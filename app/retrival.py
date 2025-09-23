# app/retrieval.py
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from app.init_pinecone import index

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using the context below.
If the answer requires combining multiple facts, reason step by step.
If the answer is not in the context, just say "I don't know".

Context:
{contexts}

Question: {question}
"""

# ----------------------------
# 1. Query Expansion
# ----------------------------
async def rewrite_query(query: str) -> str:
    """HyDE-style: generate a hypothetical passage that could answer the query"""
    prompt = f"""
    Generate a short, neutral passage that could hypothetically answer the question below.
    Do not say you don't know. Just invent a plausible answer in the style of the documents.

    Question: {query}
    """
    resp = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


async def expand_query(query: str, n: int = 3) -> list[str]:
    """Generate n alternative phrasings of the query"""
    prompt = f"""
    Rewrite the following question into {n} different alternative queries
    that capture different possible ways of asking the same thing.

    Question: {query}
    """
    resp = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    rewrites = resp.choices[0].message.content.strip().split("\n")
    # Clean and deduplicate
    rewrites = [r.strip("-• ") for r in rewrites if r.strip()]
    return list(set(rewrites))[:n]


# ----------------------------
# 2. Embeddings + Retrieval
# ----------------------------
async def embed_query(text: str) -> list[float]:
    resp = await client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


async def retrieve_docs(query: str, k: int = 5):
    """
    Expand query with HyDE + multiple rewrites, embed all,
    fetch results from Pinecone, and merge unique matches.
    """
    synthetic_query = await rewrite_query(query)
    rewrites = await expand_query(query, n=3)

    all_queries = [query, synthetic_query] + rewrites
    all_matches = []

    for q in all_queries:
        emb_q = await embed_query(q)
        res = index.query(vector=emb_q, top_k=k, include_metadata=True)
        all_matches.extend(res.matches)

    # Deduplicate results by ID, keep highest score
    unique = {}
    for m in all_matches:
        mid = m["id"]
        if mid not in unique or m["score"] > unique[mid]["score"]:
            unique[mid] = m

    # Sort by score descending
    matches = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
    return matches[:k]


# ----------------------------
# 3. Answer Generation
# ----------------------------
async def generate_answer(question: str, matches: list):
    contexts = []
    for m in matches:
        meta = m["metadata"]
        snippet = meta.get("text", "")[:800]
        contexts.append(f"Source: {meta.get('source')} (page {meta.get('page')})\n{snippet}")

    context_block = "\n\n".join(contexts)
    prompt = PROMPT_TEMPLATE.format(contexts=context_block, question=question)

    resp = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return resp.choices[0].message.content, contexts


# ----------------------------
# 4. Main Ask Function
# ----------------------------
async def ask(query: str, k: int = 10):
    print(f"❓ Question asked: {query}")
    matches = await retrieve_docs(query, k=k)
    answer, used_contexts = await generate_answer(query, matches)
    return {
        "answer": answer,
        "sources": used_contexts
    }

# Example CLI usage
if __name__ == "__main__":
    import asyncio
    from ingest import init_ingest

    init_ingest()

    # async def loop():
    #     while True:
    #         q = input("Enter your question: ")
    #         res = await ask(q)
    #         print(f"{res['answer']} \n From: {res['sources']}")

    # asyncio.run(loop())

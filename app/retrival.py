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
If the answer is not in the context, just say you don't know.

Context:
{contexts}

Question: {question}
"""

async def embed_query(text: str) -> list[float]:
    """Create an embedding for the query using OpenAI embeddings"""
    resp = await client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

async def retrieve_docs(query: str, k: int = 5):
    """Embed the query and fetch top-k matches from Pinecone"""
    emb_q = await embed_query(query)
    res = index.query(vector=emb_q, top_k=k, include_metadata=True)
    return res.matches

async def generate_answer(question: str, matches: list):
    """Assemble context and call OpenAI Chat to answer"""
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

async def ask(query: str, k: int = 5):
    matches = await retrieve_docs(query, k=k)
    answer, used_contexts = await generate_answer(query, matches)
    return {
        "answer": answer,
        "sources": used_contexts
    }

# Example CLI usage
    # async def loop():
    #     while True:
    #         q = input("Enter your question: ")
    #         res = await ask(q)
    #         print(f"{res['answer']} \n From: {res['sources']}")

    # asyncio.run(loop())

from google import genai
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
MODEL = "jina-embeddings-v3"  # or jina-embeddings-v3 or v4
DIMENSION = 1024
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
oai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding_gemini(text: str) -> list[float]:
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return resp.embeddings

def get_embeddings_gemini(texts: list[str]) -> list[list[float]]:
    """Return embeddings as list[list[float]] from Gemini"""
    resp = client.models.embed_content(
        model="models/embedding-001",
        contents=texts,
    )

    embeddings = []
    for emb in resp.embeddings:
        # unwrap ContentEmbedding → list[float]
        embeddings.append(list(emb.values))
    return embeddings

def get_embeddings_jina(texts, task="retrieval.passage"):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "input": texts,
        "model": MODEL,
        "dimensions": DIMENSION,
        "task": task,
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    j = resp.json()
    # format of resp.json()["data"] typically has list of {"embedding": [...]} etc.
    embeddings = [item["embedding"] for item in j["data"]]
    return embeddings


def get_embeddings_openai(texts: list[str]) -> list[list[float]]:
    """
    Use OpenAI embeddings to embed a batch of texts.
    Returns list of embedding vectors.
    """
    resp = oai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]
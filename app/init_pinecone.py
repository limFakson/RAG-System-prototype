import os
import math
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-demo")

pc = Pinecone(api_key=PINECONE_KEY)
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

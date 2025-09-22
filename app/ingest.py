import os
import glob
import json
from pathlib import Path
import pdfplumber
from app.model_embeddings import get_embeddings_openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.init_pinecone import index  # your pinecone index object
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()

# Configs
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDED_LOG = OUTPUT_DIR / "embedded_docs.jsonl"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 16))


def load_embedded_log():
    """Load set of already-processed docs."""
    if not EMBEDDED_LOG.exists():
        return set()
    with open(EMBEDDED_LOG, "r", encoding="utf-8") as f:
        return set(json.loads(line)["doc"] for line in f)


def update_embedded_log(doc_name: str):
    """Mark a doc as processed."""
    with open(EMBEDDED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"doc": doc_name}) + "\n")


def iter_pdf_pages(skip_docs: set):
    """
    Generator over (file_path, page_number, page_text)
    Scans DOCS_DIR recursively and skips already embedded docs.
    """
    for file_path in DOCS_DIR.rglob("*.pdf"):  # recursive search
        file_name = Path(file_path).name
        if file_name in skip_docs:
            logger.info(f"⏩ Skipping {file_name}, already ingested")
            continue

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        yield file_path, page_num, text
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path}: {e}")


def chunk_sentences(text, splitter):
    return splitter.split_text(text)


def load_and_chunk(skip_docs: set):
    """
    Generator that yields chunk dicts one by one,
    skipping already processed docs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    for file_path, page_num, page_text in iter_pdf_pages(skip_docs):
        file_stem = Path(file_path).stem
        file_name = Path(file_path).name
        sub_chunks = chunk_sentences(page_text, splitter)
        for i, sub in enumerate(sub_chunks):
            yield {
                "id": f"{file_stem}_p{page_num}_c{i}",
                "text": sub.strip(),
                "metadata": {
                    "source": file_name,
                    "page": page_num,
                    "text": sub.strip()
                },
                "doc_name": file_name,  # keep track of doc for logging
            }


def batch_iterable(iterable, batch_size):
    """Yield items from a generator in fixed-size batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_and_upsert_stream(skip_docs: set):
    """
    Stream chunks → embed in batches → upsert in Pinecone.
    """
    chunk_gen = load_and_chunk(skip_docs)
    current_doc = None
    total = 0

    for batch in batch_iterable(chunk_gen, BATCH_SIZE):
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metas = [c["metadata"] for c in batch]
        doc_names = [c["doc_name"] for c in batch]

        # in batch come from same doc
        current_doc = doc_names[0]

        embeddings = get_embeddings_openai(texts)

        vectors = [
            {"id": ids[i], "values": embeddings[i], "metadata": metas[i]}
            for i in range(len(embeddings))
        ]

        index.upsert(vectors)
        total += len(vectors)
        logger.info(f"📤 Upserted {len(vectors)} vectors. Total so far: {total}")

    # mark completed doc
    if current_doc:
        update_embedded_log(current_doc)
        logger.info(f"✅ Completed ingestion for {current_doc}")


def init_ingest():
    logger.info("🚀 Starting ingestion of new documents...")
    skip_docs = load_embedded_log()
    embed_and_upsert_stream(skip_docs)
    logger.info("🎉 Ingestion completed.")

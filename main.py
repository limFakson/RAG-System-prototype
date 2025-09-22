# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
from app.ingest import init_ingest
from app.retrival import ask
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG demo with OpenAi")


# Allow all origins (for demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)

@app.on_event("startup")
def startup_event():
    # Run ingestion in background thread
    threading.Thread(target=init_ingest, daemon=True).start()

class IngestRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict = None


class QueryRequest(BaseModel):
    question: str


@app.post("/ingest")
def api_ingest(req: IngestRequest):
    try:
        # return ingest_document(req.doc_id, req.text, req.metadata)
        pass
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def api_query(req: QueryRequest):
    try:
        return await ask(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# adapter/hf_embeddings.py
import os
from dotenv import load_dotenv
import requests
import torch
from transformers import AutoTokenizer, AutoModel

load_dotenv()
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")  # get one at https://huggingface.co/settings/tokens
HF_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
# For some HF setups you can call `/embeddings` or use the inference-pipeline endpoint shown above.
# load model once at startup

# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim embeddings

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # [batch, seq_len, hidden_dim]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert list of texts into embeddings"""
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.tolist()



# def get_embedding_hf(text: str) -> list[float]:
#     """
#     Return embedding for a single text.
#     """
#     return _model.encode(text).tolist()

# def get_embeddings_hf(texts: list[str]) -> list[list[float]]:
#     """
#     Return embeddings for a list of texts.
#     """
#     return _model.encode(texts, convert_to_numpy=True).tolist()

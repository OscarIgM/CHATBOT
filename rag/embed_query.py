# rag/embed_query.py
from sentence_transformers import SentenceTransformer
import numpy as np

_model_name = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_model_name)

def embed_query(text: str) -> np.ndarray:
    """Devuelve embedding en forma (1, dim) float32, normalizado"""
    emb = _model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    if len(emb.shape) == 1:
        emb = np.expand_dims(emb, axis=0)
    return emb.astype("float32")

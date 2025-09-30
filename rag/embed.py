import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.parquet"
INDEX_FILE = PROCESSED_DIR / "index.faiss"

chunks_df = pd.read_parquet(CHUNKS_FILE)
texts = chunks_df["text"].tolist()
print(f"{len(texts)} chunks cargados.")

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

embeddings = []
for text in tqdm(texts, desc="Generando embeddings"):
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    embeddings.append(emb)

embeddings = np.vstack(embeddings).astype("float32")
print(f"Embeddings generados: {embeddings.shape}")

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"indice FAISS creado con {index.ntotal} vectores.")

faiss.write_index(index, str(INDEX_FILE))
print(f"indice guardado en {INDEX_FILE}")

# Opcional: guardar chunks con embeddings si quieres trazabilidad
# chunks_df["embedding"] = list(embeddings)
# chunks_df.to_parquet(PROCESSED_DIR / "chunks_with_emb.parquet", index=False)

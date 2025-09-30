import os
import faiss
import pandas as pd
import numpy as np

class Retriever:
    def __init__(self, index_path=None, metadata_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(base_dir)

        self.index_path = index_path or os.path.join(root_dir, "data", "processed", "index.faiss")
        self.metadata_path = metadata_path or os.path.join(root_dir, "data", "processed", "chunks.parquet")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"No se encontró el índice FAISS en {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"No se encontró el archivo de metadata en {self.metadata_path}")

        # Cargar FAISS y metadata
        self.index = faiss.read_index(self.index_path)
        self.metadata = pd.read_parquet(self.metadata_path)

    def search(self, query_embedding: np.ndarray, k: int = 3):
        """Busca los k chunks más cercanos al embedding."""
        if query_embedding is None:
            raise ValueError("query_embedding no puede ser None")

        # Forzar tipo y dimensión correctos
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, idxs = self.index.search(query_embedding, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            doc = self.metadata.iloc[i]
            results.append({
                "content": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": float(score)
            })
        return results

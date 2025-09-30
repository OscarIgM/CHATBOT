# eval/evaluate.py
import os
import time
from dotenv import load_dotenv
import pandas as pd
from rag.pipeline import RAGPipeline
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.embed_query import embed_query
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not OPENAI_API_KEY or not DEEPSEEK_API_KEY:
    raise ValueError("❌ Faltan las API keys en el archivo .env")

# Inicializar proveedores
chatgpt = ChatGPTProvider(api_key=OPENAI_API_KEY)
deepseek = DeepSeekProvider(api_key=DEEPSEEK_API_KEY)

retriever = Retriever(
    index_path=os.path.join("data", "processed", "index.faiss"),
    metadata_path=os.path.join("data", "processed", "chunks.parquet")
)

pipeline_chatgpt = RAGPipeline(llm_provider=chatgpt, retriever=retriever)
pipeline_deepseek = RAGPipeline(llm_provider=deepseek, retriever=retriever)

gold_df = pd.read_json("eval/gold_set.jsonl", lines=True)

def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Garantiza que un embedding sea 2D (1, dim) si es 1D"""
    return arr.reshape(1, -1) if arr.ndim == 1 else arr

results = []

for _, row in gold_df.iterrows():
    question = row["question"]
    answer_gold = row["answer"]

    # --- ChatGPT ---
    start_time = time.time()
    response_chatgpt = pipeline_chatgpt.run(question)
    end_time = time.time()
    latency_chatgpt = end_time - start_time

    # --- DeepSeek ---
    start_time = time.time()
    response_deepseek = pipeline_deepseek.run(question)
    end_time = time.time()
    latency_ds = end_time - start_time

    emb_gold = ensure_2d(embed_query(answer_gold))
    emb_chatgpt = ensure_2d(embed_query(response_chatgpt))
    emb_ds = ensure_2d(embed_query(response_deepseek))

    sim_chatgpt = cosine_similarity(emb_gold, emb_chatgpt)[0][0]
    sim_ds = cosine_similarity(emb_gold, emb_ds)[0][0]

    em_chatgpt = int(response_chatgpt.strip().lower() == answer_gold.strip().lower())
    em_ds = int(response_deepseek.strip().lower() == answer_gold.strip().lower())

    od_chatgpt = 1 if "no encontré" in response_chatgpt.lower() else 0
    od_ds = 1 if "no encontré" in response_deepseek.lower() else 0

    results.append({
        "question": question,
        "answer_gold": answer_gold,
        "response_chatgpt": response_chatgpt,
        "sim_chatgpt": sim_chatgpt,
        "em_chatgpt": em_chatgpt,
        "latency_chatgpt": latency_chatgpt,
        "od_chatgpt": od_chatgpt,
        "response_ds": response_deepseek,
        "sim_ds": sim_ds,
        "em_ds": em_ds,
        "latency_ds": latency_ds,
        "od_ds": od_ds
    })

df_results = pd.DataFrame(results)
df_results.to_csv("eval/evaluation_results.csv", index=False)
print("✅ Evaluación completada. Resultados en eval/evaluation_results.csv")

import os
from dotenv import load_dotenv
from rag.pipeline import RAGPipeline
from providers.chatgpt import ChatGPTProvider
from rag.retrieve import Retriever
from rag.embed_query import embed_query

def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("❌ Faltan la API key de OpenAI en el archivo .env")

    # Inicializamos proveedor LLM
    chatgpt = ChatGPTProvider(api_key=OPENAI_API_KEY)

    # Inicializamos retriever
    retriever = Retriever(
        index_path=os.path.join("data", "processed", "index.faiss"),
        metadata_path=os.path.join("data", "processed", "chunks.parquet")
    )

    pipeline = RAGPipeline(llm_provider=chatgpt, retriever=retriever)

    print("\n🔹 Debug RAG Pipeline")
    print("Escribe tu consulta (ej: '¿Qué día inicia el calendario 2025?')")

    while True:
        query = input("\nUsuario: ")
        if query.lower() in ["salir", "exit"]:
            break

        try:
            print("\n[1] Reescribiendo consulta...")
            q_rewritten = pipeline.rewrite_query(query)
            print("Consulta reescrita:", q_rewritten)

            print("\n[2] Generando embedding de la consulta...")
            q_emb = embed_query(q_rewritten)
            print("Embedding shape:", q_emb.shape)

            print("\n[3] Buscando chunks en FAISS...")
            docs = retriever.search(q_emb, k=5)
            print(f"Chunks encontrados: {len(docs)}")
            for i, d in enumerate(docs):
                print(f"\n--- Chunk {i+1} (score: {d['score']:.4f}) ---")
                print(d['content'][:300], "...")  # primeros 300 caracteres

            print("\n[4] Sintetizando respuesta con LLM...")
            if docs:
                respuesta = pipeline.synthesize(query, docs)
                respuesta = pipeline.postprocess(respuesta)
                print("\n--- Respuesta RAG ---")
                print(respuesta)
            else:
                print("⚠️ No se encontraron chunks relevantes para la consulta.")

        except Exception as e:
            print(f"\n❌ Error en el pipeline: {e}")

if __name__ == "__main__":
    main()

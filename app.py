import os
import argparse
from dotenv import load_dotenv
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.pipeline import RAGPipeline

def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    if not OPENAI_API_KEY or not DEEPSEEK_API_KEY:
        raise ValueError("❌ Faltan las API keys en el archivo .env")

    parser = argparse.ArgumentParser(description="Consulta normativa UFRO usando RAG.")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["chatgpt", "deepseek"],
        default="chatgpt",
        help="Selecciona el proveedor de LLM a usar."
    )
    args = parser.parse_args()

    print("✅ Keys cargadas correctamente")
    print(f"🔹 Proveedor seleccionado: {args.provider}")

    chatgpt = ChatGPTProvider(api_key=OPENAI_API_KEY, temperature=0.5)
    deepseek = DeepSeekProvider(api_key=DEEPSEEK_API_KEY)

    retriever = Retriever(
        index_path=os.path.join("data", "processed", "index.faiss"),
        metadata_path=os.path.join("data", "processed", "chunks.parquet")
    )

    if args.provider == "chatgpt":
        pipeline = RAGPipeline(llm_provider=chatgpt, retriever=retriever)
    else:
        pipeline = RAGPipeline(llm_provider=deepseek, retriever=retriever)

    print("\nEscribe tu consulta sobre normativa UFRO (escribe 'salir' para terminar):\n")
    while True:
        query = input("Usuario: ")
        if query.lower() in ["salir", "exit"]:
            break

        try:
            respuesta = pipeline.run(query)
            print(f"\n--- Respuesta {args.provider.upper()} RAG ---")
            print(respuesta)

        except Exception as e:
            print(f"\n❌ Error procesando la consulta: {e}")


if __name__ == "__main__":
    main()

from rag.prompts import SYSTEM_REWRITE, build_synthesis_prompt
from rag.embed_query import embed_query

class RAGPipeline:
    def __init__(self, llm_provider, retriever):
        self.llm = llm_provider
        self.retriever = retriever

    def rewrite_query(self, query: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_REWRITE},
            {"role": "user", "content": query}
        ]
        return self.llm.chat(messages)

    def retrieve(self, query: str, k: int = 3):
        query_embedding = embed_query(query)
        return self.retriever.search(query_embedding, k)

    def synthesize(self, query: str, docs: list):
        messages = build_synthesis_prompt(query, docs)
        return self.llm.chat(messages)

    def postprocess(self, response: str) -> str:
        if "[Fuente" not in response:
            response += "\n\n(No se encontraron citas relevantes)"
        return response

    def run(self, query: str, k: int = 3):
        try:
            print("[1] Reescribiendo consulta...")
            q_rewritten = self.rewrite_query(query)
            print(f"Consulta reescrita: {q_rewritten}")

            print("[2] Generando embedding de la consulta...")
            query_embedding = embed_query(q_rewritten)
            print(f"Embedding shape: {query_embedding.shape}")

            print("[3] Buscando chunks en FAISS...")
            docs = self.retriever.search(query_embedding, k)
            print("Chunks encontrados:")
            for i, doc in enumerate(docs):
                print(f"[{i+1}] ({doc['score']:.3f}) {doc['content'][:100]}...")

            if not docs:
                return "(No se encontraron documentos relevantes)"

            print(f"[4] Sintetizando respuesta con LLM...")
            raw_answer = self.synthesize(q_rewritten, docs)
            return self.postprocess(raw_answer)

        except Exception as e:
            return f"❌ Error en el pipeline: {e}"

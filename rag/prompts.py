# rag/prompts.py

SYSTEM_REWRITE = """Reformula la pregunta del usuario en un estilo claro y breve para mejorar la búsqueda documental.
No cambies el significado de la pregunta."""

SYSTEM_SYNTHESIS = """Eres un asistente experto en normativa de la Universidad de La Frontera (UFRO).
Solo responde usando la información entregada en las fuentes.
No uses conocimientos previos.
Si la respuesta no está en las fuentes, responde exactamente: "No encontré normativa relacionada".
Incluye citas en el formato [Fuente X] siempre que sea posible.
"""

def build_synthesis_prompt(query: str, docs: list):
    context = "\n\n".join([f"[Fuente {i+1}] {doc['content']}" for i, doc in enumerate(docs)])
    return [
        {"role": "system", "content": SYSTEM_SYNTHESIS},
        {"role": "user", "content": f"Pregunta: {query}\n\nFuentes:\n{context}"}
    ]

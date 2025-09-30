import os
from dotenv import load_dotenv
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider

def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    if not OPENAI_API_KEY or not DEEPSEEK_API_KEY:
        raise ValueError("❌ Faltan las API keys en el archivo .env")

    print("✅ Keys cargadas correctamente")

    # Pasamos la API key al constructor
    chatgpt = ChatGPTProvider(api_key=OPENAI_API_KEY)
    deepseek = DeepSeekProvider(api_key=DEEPSEEK_API_KEY)

    print("\nEscribe tu consulta sobre normativa UFRO (escribe 'salir' para terminar):\n")
    while True:
        query = input("Usuario: ")
        if query.lower() in ["salir", "exit"]:
            break

        messages = [
            {"role": "system", "content": "Actúa como un asistente experto en normativa de la Universidad de La Frontera. Responde de manera clara y concisa."},
            {"role": "user", "content": f"Pregunta del usuario: {query}"}
        ]

        response_chatgpt = chatgpt.chat(messages)
        response_deepseek = deepseek.chat(messages)

        print("\n--- Respuesta ChatGPT ---")
        print(response_chatgpt)
        print("\n--- Respuesta DeepSeek ---")
        print(response_deepseek)
   

if __name__ == "__main__":
    main()

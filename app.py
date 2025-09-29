import os
from dotenv import load_dotenv

def main():
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    print(f" - OPENAI_API_KEY: {'OK' if openai_key else 'no encontrada'}")
    print(f" - DEEPSEEK_API_KEY: {'OK' if deepseek_key else 'no encontrada'}")

    print("\n Entorno funcionando")

if __name__ == "__main__":
    main()

from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.pipeline import RAGPipeline

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

chatgpt = ChatGPTProvider(api_key=OPENAI_API_KEY, temperature=0.5)
deepseek = DeepSeekProvider(api_key=DEEPSEEK_API_KEY)

retriever = Retriever(
    index_path=os.path.join("data", "processed", "index.faiss"),
    metadata_path=os.path.join("data", "processed", "chunks.parquet")
)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # HTML simple con un form

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    provider_name = data.get("provider", "chatgpt")

    provider = chatgpt if provider_name == "chatgpt" else deepseek
    pipeline = RAGPipeline(llm_provider=provider, retriever=retriever)
    answer = pipeline.run(query)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

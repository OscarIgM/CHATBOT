import requests
from typing import List, Dict
from .base import Provider

class DeepSeekProvider(Provider):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self._model = model
        self._name = f"DeepSeek ({self._model})"
        self.base_url = "https://api.deepseek.com/v1"

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        model = kwargs.pop("model", self._model)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error en DeepSeek: {e}")
            return "ERROR: Falló la comunicación con DeepSeek."

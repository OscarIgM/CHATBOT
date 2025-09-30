from abc import ABC, abstractmethod
from typing import List, Dict
class Provider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del proveedor"""
        pass

@abstractmethod
def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Envía un mensaje al LLM y obtiene la respuesta.

    :param messages: Historial de mensajes (incluye el system prompt).
    :param kwargs: Parámetros adicionales (e.g., temperatura, model_name).
    :return: La respuesta de texto generada.
    """
    pass
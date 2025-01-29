"""
OLLAMA provider.
"""

import logging

import ollama

from src.generic_model import Model


def _verify_model(name: str) -> str:
    models = ollama.list()["models"]
    if ":" not in name:
        name = name + ":latest"
    matching_models = [m for m in models if _matches_model(m, name)]
    if len(matching_models) == 0:
        logging.error(
            "Model not found: %s, installed models: %s",
            name,
            ", ".join([m["model"] for m in models]),
        )
        return None
    logging.info("Will use OLLAMA model: %s", matching_models[0])
    return name


def _matches_model(
    model: ollama.ListResponse.Model,
    name: str,
) -> bool:
    if model["model"] == name:
        return True
    return False


class OllamaModel(Model):
    """
    OLLAMA model.
    """

    def __init__(self, model_name):
        self.name = _verify_model(model_name)
        if self.name is None:
            return

    def chat(self, messages) -> str:
        """
        Sends messages to the model and returns the response.
        """
        response = ollama.chat(
            model=self.name,
            messages=messages,
        )
        return response["message"]["content"]

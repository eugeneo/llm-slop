"""
This module contains the main entry point for the application.
"""

import logging
import time
from typing import List, Text

import ollama
from absl import app, flags
from dotenv import load_dotenv
from yaml import FullLoader, load

from src.generic_model import Model
from src.ollama_provider import OllamaModel

load_dotenv()

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "deepseek-r1:8b", "Model to use for chat.")
flags.DEFINE_string("output", None, "File to write output to.")

logging.basicConfig(level=logging.DEBUG)


def _init_model(model: str) -> Model:
    return OllamaModel(model)


def _sanitize(message: dict) -> ollama.Message:
    if "content" not in message:
        message["content"] = ""
    if "role" not in message:
        message["role"] = "user"
    return message


def _read_messages(files: List[str]) -> List[ollama.Message]:
    messages = []
    for file in files:
        logging.info("Parsing %s", file)
        try:
            with open(file, "r", encoding="utf-8") as file:
                content = file.read()
                messages.extend(
                    [_sanitize(m) for m in load(content, Loader=FullLoader)]
                )

        except FileNotFoundError:
            logging.error("File not found: %s", file)
            return []
    return messages


def _write_output(content: str, file: str | None, t: float) -> None:
    if FLAGS.output is not None:
        try:
            with open(FLAGS.output, "w", encoding="utf-8") as file:
                file.write(content)
                logging.info("Wrote response to %s", FLAGS.output)
        except OSError:
            logging.info("Received response %s", content)
            raise
    else:
        logging.info("Received response %s", content)
    logging.info("Time taken: %ds", t)


def _main(args: List[Text]) -> None:
    if len(args) < 2:
        logging.error("No arguments provided.")
        return
    model = _init_model(FLAGS.model)
    messages = _read_messages(args[1:])
    logging.info("Sending message to Ollama: %s", messages)
    start = time.time()
    output = model.chat(messages)
    _write_output(output, FLAGS.output, time.time() - start)


if __name__ == "__main__":
    app.run(_main)

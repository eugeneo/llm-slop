"""
This module contains the main entry point for the application.
"""

import logging
from typing import List, Text

from absl import app, flags
import ollama
from yaml import load, FullLoader

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "deepseek-r1:8b", "Model to use for chat.")
flags.DEFINE_string("output", None, "File to write output to.")

logging.basicConfig(level=logging.DEBUG)


def _sanitize(message: dict) -> ollama.Message:
    if "content" not in message:
        message["content"] = ""
    if "role" not in message:
        message["role"] = "user"
    return message


def _matches_model(
    model: ollama.ListResponse.Model,
    name: str,
) -> bool:
    if model["model"] == name:
        return True
    return False


def _verify_model(name: str) -> str:
    models = ollama.list()["models"]
    if ":" not in name:
        name = name + ":latest"
    matching_models = [m for m in models if _matches_model(m, name)]
    if len(matching_models) == 0:
        logging.error(
            "Model not found: %s, installed models: %s",
            FLAGS.model,
            ", ".join([m["model"] for m in models]),
        )
        return None
    logging.info("Will use OLLAMA model: %s", matching_models[0])
    return name


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


def _write_output(content: str, file: str | None) -> None:
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


def _main(args: List[Text]) -> None:
    if len(args) < 2:
        logging.error("No arguments provided.")
        return
    name = _verify_model(FLAGS.model)
    if name is None:
        return
    messages = _read_messages(args[1:])
    logging.info("Sending message to Ollama: %s", messages)
    response = ollama.chat(
        model=name,
        messages=messages,
    )
    _write_output(response["message"]["content"], FLAGS.output)


if __name__ == "__main__":
    app.run(_main)

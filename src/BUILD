load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "generic_model",
    srcs = ["generic_model.py"],
    deps = [],
)

py_library(
    name = "ollama_provider",
    srcs = ["ollama_provider.py"],
    deps = [
        ":generic_model",
        "@abseil-py//absl/flags",
        "@pypi//ollama:pkg",
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":generic_model",
        ":ollama_provider",
        "@abseil-py//absl:app",
        "@pypi//ollama:pkg",
        "@pypi//python_dotenv:pkg",
        "@pypi//pyyaml:pkg",
    ],
)

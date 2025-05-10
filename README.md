# llm-llama-server

[![PyPI](https://img.shields.io/pypi/v/llm-llama-server.svg)](https://pypi.org/project/llm-llama-server/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-llama-server?include_prereleases&label=changelog)](https://github.com/simonw/llm-llama-server/releases)
[![Tests](https://github.com/simonw/llm-llama-server/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-llama-server/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-llama-server/blob/main/LICENSE)

Interact with llama-server models

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-llama-server
```
## Usage

You'll need to be running a [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) on port 8080 to use this plugin.

To access a regular model from LLM, use the `llama-server` model:
```bash
llm -m llama-server "say hi"
```
For vision models, use `llama-server-vision`:
```bash
llm -m llama-server-vision describe -a path/to/image.png
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-llama-server
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
python -m pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```

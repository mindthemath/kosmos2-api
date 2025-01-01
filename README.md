[![Build and Push Docker Image](https://github.com/mindthemath/kosmos-2-api/actions/workflows/docker.yml/badge.svg)](https://github.com/mindthemath/kosmos-2-api/actions/workflows/docker.yml)
[![Docker Image](https://img.shields.io/docker/v/mindthemath/kosmos2-api/latest)](https://hub.docker.com/r/mindthemath/kosmos2-api/tags)

# Kosmos-2 (as an API)

* Kosmos-2 is a Multimodal Large Language Model (MLLM) that enables new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world.

* It represents refer expressions as links in Markdown, using the format [text span](bounding boxes), where object descriptions are sequences of location tokens.

* Markdown is a convenient file format for writing and editing text, and is easily converted to HTML, or adapted for downstream use in other LLM-related tasks, as this is a format that should be well-represented in their training data.

* This repository exposes the `microsoft/kosmos-2-patch14-224` checkpoint from Hugging Face Hub via [litserve](https://github.com/lightning-ai/litserve) as an API endpoint at `/predict`.

For more information, refer to the [arxiv](https://arxiv.org/abs/2306.14824) paper.


## Example Usage

Makefile: `test` will download and send a test image to the server.
Ensure you have `jq` installed for parsing the JSON output, and `curl` for downloading the image + testing the endpoint.

```bash
curl -fsSL https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png -o snowman.png
```

```bash
curl -X POST -F "content=@snowman.png" http://127.0.0.1:8000/predict | jq '.output'
```

(replace the URL for the endpoint if deploying elsewhere).

You can add `-F "prompt=<your prompt>"` as an additional field to perform other tasks with the model. The default prompt is `"<grounding> Describe this image:"`.

You can override the default by using the environment variable `DEFAULT_PROMPT` passed to the `docker run` command.

## Environment Variables

All except the first and last are `litserve`-specific.

```python
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
PORT = int(os.environ.get("PORT", "8000"))
NUM_API_SERVERS = int(os.environ.get("NUM_API_SERVERS", "1"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "2"))
DEFAULT_PROMPT = os.environ.get("DEFAULT_PROMPT", "<grounding> Describe this image:")
```


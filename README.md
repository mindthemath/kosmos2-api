# Kosmos-2 API

[Kosmos-2](https://github.com/microsoft/unilm/tree/master/kosmos-2) is:
> a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., ``[text span](bounding boxes)'', where object descriptions are sequences of location tokens.
> 
> -- [arxiv](https://arxiv.org/abs/2306.14824)

This repository exposes the `microsoft/kosmos-2-patch14-224` checkpoint from hugging face hub via [litserve](https://github.com/lightning-ai/litserve) as an API endpoint at `/predict`.


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

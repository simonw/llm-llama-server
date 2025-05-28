import httpx
import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
import os


class LlamaServer(Chat):
    model_id = "llama-server"
    key = "sk-llama-server"

    def __init__(self, **kwargs):
        super().__init__(
            model_name="llama-server",
            model_id=self.model_id,
            api_base="http://localhost:8080/v1",
            **kwargs,
        )

    def execute(self, prompt, stream, response, conversation=None, key=None):
        yield from super().execute(prompt, stream, response, conversation, key)
        # Quick timeout limited hit to get resolved_model_id
        try:
            http_response = httpx.get(
                f"{self.api_base}/models",
                timeout=httpx.Timeout(0.1, connect=0.1),
            )
            http_response.raise_for_status()
            set_resolved_model(response, http_response.json())
        except httpx.HTTPError:
            pass

    def __str__(self):
        return "llama-server: {}".format(self.model_id)


class AsyncLlamaServer(AsyncChat):
    model_id = "llama-server"
    key = "sk-llama-server"

    def __init__(self, **kwargs):
        super().__init__(
            model_name="llama-server",
            model_id=self.model_id,
            api_base="http://localhost:8080/v1",
            **kwargs,
        )

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        async for chunk in super().execute(prompt, stream, response, conversation, key):
            yield chunk
        try:
            async with httpx.AsyncClient() as client:
                http_response = await client.get(
                    f"{self.api_base}/models",
                    timeout=httpx.Timeout(0.1, connect=0.1),
                )
            http_response.raise_for_status()
            set_resolved_model(response, http_response.json())
        except httpx.HTTPError:
            raise

    def __str__(self):
        return f"llama-server (async): {self.model_id}"


class LlamaServerVision(LlamaServer):
    model_id = "llama-server-vision"


class AsyncLlamaServerVision(AsyncLlamaServer):
    model_id = "llama-server-vision"


class LlamaServerTools(LlamaServer):
    model_id = "llama-server-tools"


class AsyncLlamaServerTools(AsyncLlamaServer):
    model_id = "llama-server-tools"


def set_resolved_model(response, data):
    try:
        model_path = data["data"][0]["id"]
        # This will be something like:
        # '.../Caches/llama.cpp/unsloth_gemma-3-12b-it-qat-GGUF_gemma-3-12b-it-qat-Q4_K_M.gguf'
        resolved_model = os.path.basename(model_path)
        response.set_resolved_model(resolved_model)
    except (IndexError, KeyError):
        raise


@llm.hookimpl
def register_models(register):
    register(
        LlamaServer(),
        AsyncLlamaServer(),
    )
    register(
        LlamaServerVision(vision=True),
        AsyncLlamaServerVision(vision=True),
    )
    register(
        LlamaServerTools(vision=True, can_stream=False, supports_tools=True),
        AsyncLlamaServerTools(vision=True, can_stream=False, supports_tools=True),
    )

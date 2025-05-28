import llm
from llm.default_plugins.openai_models import Chat, AsyncChat


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

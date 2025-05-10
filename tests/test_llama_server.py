from llm import get_models, get_async_models


def test_plugin_is_installed():
    models = [model.model_id for model in get_models()]
    async_models = [model.model_id for model in get_async_models()]
    assert "llama-server" in models
    assert "llama-server-vision" in models
    assert "llama-server" in async_models
    assert "llama-server-vision" in async_models

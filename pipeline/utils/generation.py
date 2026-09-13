"""Model-specific generation settings shared by extraction and enrichment."""


QWEN_FLASH_MODELS = frozenset({
    "qwen/qwen3.8-flash",
    "qwen/qwen3.8-flash-next",
    "qwen/qwen3.8-flash-next-fp8",
})


def generation_kwargs(model_name: str, endpoint: str) -> dict[str, object]:
    """Qwen's recommended thinking-mode settings; no overrides for other models.

    Sources: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
    https://openrouter.ai/api/v1/models/qwen/qwen3.8-flash/endpoints
    Alibaba does not advertise min_p or repetition_penalty support. Their
    recommended neutral values are sent only to the local vLLM endpoint.
    """
    if model_name.casefold() not in QWEN_FLASH_MODELS:
        return {}
    extra_body: dict[str, object] = {"top_k": 20}
    if endpoint == "openrouter":
        extra_body.update({
            "reasoning": {"enabled": True},
            "provider": {
                "only": ["alibaba"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        })
    else:
        extra_body.update({
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        })
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": extra_body,
    }


def make_profiled_langextract_model(
    model_name: str, endpoint: str, connection: dict[str, str], max_workers: int
):
    """Preserve request extensions that LangExtract's provider filters out.

    Override its request builder so routing survives both ordinary requests and
    batch requests, while the base provider still owns schema construction.
    """
    from langextract.providers.openai import OpenAILanguageModel

    class ProfiledOpenAILanguageModel(OpenAILanguageModel):
        def _build_chat_completions_params(self, prompt: str, config: dict) -> dict:
            params = super()._build_chat_completions_params(prompt, config)
            params.update(generation_kwargs(model_name, endpoint))
            return params

    return ProfiledOpenAILanguageModel(
        model_id=model_name, max_workers=max_workers, **connection
    )

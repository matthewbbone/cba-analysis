import json
from json import JSONDecodeError
import os
import queue
import re
from contextlib import contextmanager

from openai import OpenAI


MODEL_PRICING = {
    "gpt-5.5": {
        "input": 5.00,
        "cached_input": 0.50,
        "output": 30.00,
    },
    "gpt-5.4": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
    },
    "gpt-5.4-mini": {
        "input": 0.75,
        "cached_input": 0.075,
        "output": 4.50,
    },
    "gpt-5.4-nano": {
        "input": 0.20,
        "cached_input": 0.02,
        "output": 1.25,
    },
    "qwen/qwen3.6-35b-a3b": {
        "input": 0.1612,
        "cached_input": 0.0,
        "output": 0.9653,
    },
}


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", model_name.lower())
    return re.sub(r"_+", "_", slug).strip("_")


def build_query(model_name: str, system_prompt: str, prompt: str, schema: dict) -> dict:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "response_format": schema,
    }


def _get_value(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cost(tokens, price_per_million):
    return (tokens / 1_000_000) * price_per_million


def extract_usage_and_cost(response, model_name):
    if model_name not in MODEL_PRICING:
        raise ValueError(f"No LLM pricing configured for model: {model_name}")

    pricing = MODEL_PRICING[model_name]
    usage = _get_value(response, "usage")
    prompt_details = _get_value(usage, "prompt_tokens_details")
    completion_details = _get_value(usage, "completion_tokens_details")

    input_tokens = _get_value(usage, "prompt_tokens", 0) or 0
    output_tokens = _get_value(usage, "completion_tokens", 0) or 0
    total_tokens = _get_value(usage, "total_tokens", input_tokens + output_tokens) or 0
    cached_input_tokens = _get_value(prompt_details, "cached_tokens", 0) or 0
    reasoning_tokens = _get_value(completion_details, "reasoning_tokens", 0) or 0

    billable_input_tokens = max(input_tokens - cached_input_tokens, 0)
    visible_output_tokens = max(output_tokens - reasoning_tokens, 0)

    input_cost_usd = _cost(billable_input_tokens, pricing["input"])
    cached_input_cost_usd = _cost(cached_input_tokens, pricing["cached_input"])
    output_cost_usd = _cost(output_tokens, pricing["output"])

    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "billable_input_tokens": billable_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_output_tokens": visible_output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost_usd,
        "cached_input_cost_usd": cached_input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "total_cost_usd": input_cost_usd + cached_input_cost_usd + output_cost_usd,
    }


def create_llm_client(model_name: str) -> OpenAI:
    if "gpt" in model_name:
        return OpenAI()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for non-GPT models.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


class LLMClient:
    def __init__(self, model_name: str, max_json_retries: int = 2):
        self.model_name = model_name
        self.max_json_retries = max_json_retries
        self.client = create_llm_client(model_name)

    def call_json(self, system_prompt: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        last_error = None
        last_raw = ""
        last_usage = None
        last_finish_reason = None

        for attempt in range(self.max_json_retries + 1):
            retry_prompt = prompt
            if attempt:
                retry_prompt = " ".join(
                    [
                        prompt,
                        "\n\nYour previous response was not valid JSON.",
                        "Return only valid JSON that matches the required schema.",
                        "Do not include markdown fences or explanatory text.",
                    ]
                )

            query = build_query(self.model_name, system_prompt, retry_prompt, schema)
            response = self.client.chat.completions.create(**query)
            usage = extract_usage_and_cost(response, self.model_name)
            choice = response.choices[0]
            raw = choice.message.content or ""

            try:
                payload = json.loads(raw) if raw.strip() else {}
                return payload, usage
            except JSONDecodeError as exc:
                last_error = exc
                last_raw = raw
                last_usage = usage
                last_finish_reason = getattr(choice, "finish_reason", None)

        preview = last_raw[:500].replace("\n", "\\n")
        error = ValueError(
            "LLM returned invalid JSON after "
            f"{self.max_json_retries + 1} attempt(s); "
            f"finish_reason={last_finish_reason!r}; "
            f"usage={last_usage}; raw_preview={preview!r}"
        )
        raise error from last_error


class LLMClientPool:
    def __init__(self, model_name: str, size: int, max_json_retries: int = 2):
        if size < 1:
            raise ValueError("LLMClientPool size must be at least 1.")
        self.model_name = model_name
        self.size = size
        self._clients = queue.Queue(maxsize=size)
        for _ in range(size):
            self._clients.put(
                LLMClient(
                    model_name,
                    max_json_retries=max_json_retries,
                )
            )

    @contextmanager
    def client(self):
        llm_client = self._clients.get()
        try:
            yield llm_client
        finally:
            self._clients.put(llm_client)

    def call_json(self, system_prompt: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        with self.client() as llm_client:
            return llm_client.call_json(system_prompt, prompt, schema)

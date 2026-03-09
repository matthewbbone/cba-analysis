from __future__ import annotations

from importlib.util import find_spec


def register_qwen35_compat() -> bool:
    """Register a local alias for Qwen 3.5 MoE on transformers builds that lack it."""
    try:
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except Exception:
        return False

    if "qwen3_5_moe" in CONFIG_MAPPING_NAMES and find_spec("transformers.models.qwen3_5_moe") is not None:
        return True

    if CONFIG_MAPPING_NAMES.get("qwen3_moe") is None:
        return False

    try:
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    except Exception:
        return False

    class Qwen35MoeConfig(Qwen3MoeConfig):
        model_type = "qwen3_5_moe"

    AutoConfig.register("qwen3_5_moe", Qwen35MoeConfig, exist_ok=True)
    return True

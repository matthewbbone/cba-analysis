import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def path_safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def default_cache_dir() -> Path:
    return resolve_project_path(os.environ.get("CACHE_DIR", "cache"))

"""Shared CUDA device-selection helpers for model runner CLIs."""

from __future__ import annotations

import re


CUDA_DEVICE_ID_PATTERN = re.compile(r"[0-9]+")


def normalize_cuda_device_ids(device: str | int | None) -> str | None:
    """Return a canonical CUDA device list or raise for an invalid selector."""

    if device is None:
        return None
    if isinstance(device, bool):
        raise ValueError("--device must contain non-negative integer GPU IDs")

    raw_value = str(device).strip()
    if not raw_value:
        raise ValueError("--device must contain at least one GPU ID")

    raw_ids = raw_value.split(",")
    if any(not raw_id.strip() for raw_id in raw_ids):
        raise ValueError(
            "--device must be a comma-separated list without empty GPU IDs"
        )

    device_ids: list[str] = []
    for raw_id in raw_ids:
        candidate = raw_id.strip()
        if CUDA_DEVICE_ID_PATTERN.fullmatch(candidate) is None:
            raise ValueError(
                "--device must contain only non-negative integer GPU IDs"
            )
        device_ids.append(str(int(candidate)))

    if len(set(device_ids)) != len(device_ids):
        raise ValueError("--device cannot contain duplicate GPU IDs")
    return ",".join(device_ids)


def validate_cuda_device_selection(
    device: str | int | None,
    num_gpus: int,
) -> str | None:
    """Validate and normalize a CUDA selector for a tensor-parallel width."""

    normalized = normalize_cuda_device_ids(device)
    if normalized is None:
        return None

    selected_count = len(normalized.split(","))
    if selected_count != num_gpus:
        raise ValueError(
            f"--device selects {selected_count} GPU(s), but --num-gpus is "
            f"{num_gpus}; provide exactly one device ID per requested GPU"
        )
    return normalized


__all__ = ["normalize_cuda_device_ids", "validate_cuda_device_selection"]

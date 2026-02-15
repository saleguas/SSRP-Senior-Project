from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available.")

    device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    if arch not in arch_list:
        device_name = torch.cuda.get_device_name(device_index)
        supported = ", ".join(arch_list) if arch_list else "unknown"
        raise RuntimeError(
            f"CUDA GPU detected ({device_name}, {arch}) but this PyTorch build "
            f"does not support it (supported: {supported}). Install a CUDA 12.x build."
        )


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def configure_ultralytics(models_root: Path) -> None:
    from ultralytics.utils import SETTINGS

    SETTINGS.update(weights_dir=str(models_root.resolve()))

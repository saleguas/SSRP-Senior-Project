from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([path for path in folder.iterdir() if is_image_file(path)])


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


class _TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextmanager
def tee_output(log_path: Path | None, loggers: Iterable[logging.Logger] = ()) -> Iterator[None]:
    if log_path is None:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        stdout = sys.stdout
        stderr = sys.stderr
        tee_stdout = _TeeStream(stdout, handle)
        tee_stderr = _TeeStream(stderr, handle)
        original_streams: list[tuple[logging.Handler, object]] = []
        try:
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            for logger in loggers:
                for handler in getattr(logger, "handlers", []):
                    if hasattr(handler, "stream"):
                        original_streams.append((handler, handler.stream))
                        handler.setStream(tee_stdout)
            yield
        finally:
            for handler, stream in original_streams:
                handler.setStream(stream)
            sys.stdout = stdout
            sys.stderr = stderr

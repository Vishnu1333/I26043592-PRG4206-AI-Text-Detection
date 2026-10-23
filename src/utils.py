"""
src/utils.py — Shared utilities: logging, seeding, I/O helpers.
PRG4206 | AI-Generated Text Detection System
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger.

    Args:
        name:  Module name (typically ``__name__``).
        level: Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility.

    Args:
        seed: Integer random seed. Default ``42``.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed — silently skip


# ── I/O helpers ──────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path, guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    """Serialise *data* to a JSON file.

    Args:
        data: JSON-serialisable object.
        path: Destination file path (parent directories created automatically).
    """
    import json

    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load a JSON file and return the parsed object.

    Args:
        path: Source file path.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    import json

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

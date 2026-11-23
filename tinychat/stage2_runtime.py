"""Workspace resource guards shared by the opt-in Stage 2 pipeline."""
from __future__ import annotations

import math
from pathlib import Path
import shutil

GIB = 1024 ** 3


class DiskBudgetError(RuntimeError):
    """An operation would use the user's reserved disk space."""


def ensure_disk_budget(path, required_bytes=0, min_free_gib=30.0, safety_gib=2.0):
    """Require reserve + safety + the operation's worst-case extra allocation.

    Call before downloading/writing chunks and before atomic checkpoint writes.
    The safety allowance remains available for orderly stopping and metadata.
    Existing targets are not credited: atomic replacement needs a temporary copy.
    Returns current free bytes; never deletes files or changes other drives.
    """
    if isinstance(required_bytes, bool) or not isinstance(required_bytes, int) or required_bytes < 0:
        raise ValueError("required_bytes must be a nonnegative integer")
    for name, value in (("min_free_gib", min_free_gib), ("safety_gib", safety_gib)):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    target = Path(path).resolve()
    while not target.exists():
        if target.parent == target:
            raise FileNotFoundError(path)
        target = target.parent
    if target.is_file():
        target = target.parent
    free = shutil.disk_usage(target).free
    threshold = math.ceil((min_free_gib + safety_gib) * GIB) + required_bytes
    if free < threshold:
        raise DiskBudgetError(
            f"Disk reserve: {free / GIB:.3f} GiB free at {target}; "
            f"need {threshold / GIB:.3f} GiB before allocation "
            f"({min_free_gib:g} GiB protected + {safety_gib:g} GiB safety "
            f"+ {required_bytes / GIB:.3f} GiB write). Stop before writing."
        )
    return free

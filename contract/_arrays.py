"""Array checks shared by the v2 contract types.

Every array a v2 noun holds is checked once, when the noun is constructed:
exact dtype, exact shape, then made read-only. Nothing is coerced: a backend
that hands over the wrong dtype or shape is told so, by field name, at the
seam.
"""
from __future__ import annotations

import numpy as np


def check_and_freeze(
    name: str, arr: np.ndarray, dtype: type, shape: tuple[int, ...]
) -> None:
    """Check one array field's dtype and shape, then make it read-only.

    The array is frozen in place: the caller's own reference becomes
    read-only too, so a backend hands over arrays it will not write again.

    Args:
        name: The field's name, used in the error message.
        arr: The array to check.
        dtype: The exact dtype required, e.g. np.int32.
        shape: The exact shape required.

    Raises:
        ValueError: If arr is not a numpy array, or its dtype or shape
            differs from what is required.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name}: expected a numpy array, got {type(arr).__name__}")
    if arr.dtype != np.dtype(dtype):
        raise ValueError(f"{name}: dtype {arr.dtype}, expected {np.dtype(dtype)}")
    if arr.shape != shape:
        raise ValueError(f"{name}: shape {arr.shape}, expected {shape}")
    arr.flags.writeable = False
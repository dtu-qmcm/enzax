"""General-purpose helpers that do not belong to any one part of enzax."""

from collections.abc import Sequence

import numpy as np

from enzax.array_types import FrozenArray


def pad_rows(
    rows: Sequence[Sequence[int | float] | np.ndarray],
    fill: int | float,
    dtype: type = int,
) -> np.ndarray:
    """Stack ragged rows into a matrix, padding short rows with `fill`.

    The matrix has at least one column, so that it can be indexed even when
    every row is empty.
    """
    width = max([1, *(len(row) for row in rows)])
    out = np.full((len(rows), width), fill, dtype=dtype)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def freeze_array(values) -> FrozenArray:
    """Put an array into a form that a static field can hold.

    This is required because static fields need to be comparable with ==, which
    numpy arrays are not. See https://github.com/dtu-qmcm/enzax/issues/65.
    """
    array = np.asarray(values)
    return array.shape, tuple(array.ravel().tolist())


def unfreeze_array(frozen: FrozenArray, dtype) -> np.ndarray:
    """Turn a frozen array into a numpy array with the given dtype."""
    shape, values = frozen
    return np.array(values, dtype=dtype).reshape(shape)


def get_ix_from_list(s: str, list_of_strings: list[str]):
    return next(i for i, si in enumerate(list_of_strings) if si == s)

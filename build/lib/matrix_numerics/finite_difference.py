"""Backend-neutral local polynomial derivative fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


# Shared production contract. Central differences are diagnostic/analysis
# choices and are never selected implicitly by a MATRIX tool.
DEFAULT_FINITE_DIFFERENCE_STENCIL = "forward"
ALLOWED_FINITE_DIFFERENCE_STENCILS = frozenset({"forward", "central"})


def normalize_finite_difference_stencil(stencil: str) -> str:
    """Validate and normalize the shared finite-difference stencil policy."""

    value = str(stencil).strip().casefold().replace("_", "-")
    if value not in ALLOWED_FINITE_DIFFERENCE_STENCILS:
        allowed = ", ".join(sorted(ALLOWED_FINITE_DIFFERENCE_STENCILS))
        raise ValueError(f"finite-difference stencil must be one of: {allowed}")
    return value


@dataclass(frozen=True)
class TaylorDerivativeFit:
    """Derivatives obtained from a factorial-scaled local Taylor fit."""

    derivatives: tuple[np.ndarray, ...]
    rank: int
    residual_norm: float


def fit_taylor_derivatives(
    displacements: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    max_order: int = 4,
) -> TaylorDerivativeFit:
    """Fit derivatives at zero for scalar or array-valued observations."""

    x = np.asarray(displacements, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError("at least two points are needed for finite differences")
    if not np.all(np.isfinite(x)):
        raise ValueError("finite-difference displacements must be finite")
    if np.unique(x).size != x.size:
        raise ValueError("finite-difference displacements must be unique")

    observed = np.asarray(values, dtype=float)
    if observed.ndim == 0 or observed.shape[0] != x.size:
        raise ValueError("values must have one leading entry per displacement")
    if not np.all(np.isfinite(observed)):
        raise ValueError("finite-difference values must be finite")

    order = max(1, min(int(max_order), x.size - 1))
    design = np.column_stack(
        [
            np.power(x, power) / float(math.factorial(power))
            for power in range(order + 1)
        ]
    )
    value_shape = observed.shape[1:]
    flattened = observed.reshape(x.size, -1)
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        design, flattened, rcond=None
    )
    fitted = design @ coefficients
    residual_norm = float(np.linalg.norm(flattened - fitted))
    derivatives = tuple(
        np.asarray(row, dtype=float).reshape(value_shape)
        for row in coefficients[1:]
    )
    return TaylorDerivativeFit(
        derivatives=derivatives,
        rank=int(rank),
        residual_norm=residual_norm,
    )

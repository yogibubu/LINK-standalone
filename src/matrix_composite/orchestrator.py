"""Backend-independent composite-method orchestration.

The orchestrator knows only signed terms and returned properties.  Backend
execution, coordinate representations, and finite-difference policies remain
outside this package, so the same object can be used by geometry optimizers,
single-point workflows, and mixed-program protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompositeTerm:
    """A signed contribution to a composite energy or derivative."""

    label: str
    coefficient: float
    backend: Any = None

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("a composite term needs a non-empty label")
        coefficient = float(self.coefficient)
        if not np.isfinite(coefficient) or coefficient == 0.0:
            raise ValueError("a composite coefficient must be finite and non-zero")
        object.__setattr__(self, "label", str(self.label).strip())
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class CompositeTermResult:
    """Energy and optional derivatives returned for one composite term."""

    term: CompositeTerm
    energy: float
    gradient: np.ndarray | None = None
    hessian: np.ndarray | None = None
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompositeAssembly:
    """Signed assembly result with explicit derivative completeness."""

    energy: float
    gradient: np.ndarray | None
    hessian: np.ndarray | None
    terms: tuple[CompositeTermResult, ...]
    missing_gradient_terms: tuple[str, ...]
    missing_hessian_terms: tuple[str, ...]

    @property
    def energy_evaluations(self) -> int:
        return len(self.terms)

    @property
    def gradient_evaluations(self) -> int:
        return sum(item.gradient is not None for item in self.terms)

    @property
    def hessian_evaluations(self) -> int:
        return sum(item.hessian is not None for item in self.terms)


class CompositeOrchestrator:
    """Evaluate and assemble arbitrary signed multi-backend contributions."""

    def __init__(self, terms: Sequence[CompositeTerm]) -> None:
        self.terms = tuple(terms)
        if not self.terms:
            raise ValueError("a composite orchestrator needs at least one term")

    def evaluate(
        self,
        evaluator: Callable[[CompositeTerm], CompositeTermResult],
    ) -> CompositeAssembly:
        results = tuple(evaluator(term) for term in self.terms)
        return self.assemble(results)

    def evaluate_hybrid(
        self,
        evaluator: Callable[[CompositeTerm], CompositeTermResult],
        *,
        numerical_gradient: Callable[[CompositeTerm], np.ndarray] | None = None,
        numerical_hessian: Callable[[CompositeTerm], np.ndarray] | None = None,
    ) -> CompositeAssembly:
        """Assemble analytic terms and numerically supplied missing terms.

        The callbacks are deliberately term-level: a caller may differentiate
        a missing contribution in any coordinate space while retaining
        analytic derivatives from the other terms.  The orchestrator does not
        assume Cartesian coordinates, a particular finite-difference stencil,
        or a particular backend.
        """

        results: list[CompositeTermResult] = []
        for term in self.terms:
            result = evaluator(term)
            gradient = result.gradient
            hessian = result.hessian
            if gradient is None and numerical_gradient is not None:
                gradient = np.asarray(numerical_gradient(term), dtype=float)
            if hessian is None and numerical_hessian is not None:
                hessian = np.asarray(numerical_hessian(term), dtype=float)
            results.append(
                CompositeTermResult(
                    term=term,
                    energy=result.energy,
                    gradient=gradient,
                    hessian=hessian,
                    provenance=result.provenance,
                )
            )
        return self.assemble(results)

    @staticmethod
    def assemble(results: Sequence[CompositeTermResult]) -> CompositeAssembly:
        terms = tuple(results)
        if not terms:
            raise ValueError("cannot assemble an empty composite result")
        energy = float(sum(item.term.coefficient * item.energy for item in terms))
        gradient = _assemble_array(terms, "gradient")
        hessian = _assemble_array(terms, "hessian")
        return CompositeAssembly(
            energy=energy,
            gradient=gradient,
            hessian=hessian,
            terms=terms,
            missing_gradient_terms=tuple(
                item.term.label for item in terms if item.gradient is None
            ),
            missing_hessian_terms=tuple(
                item.term.label for item in terms if item.hessian is None
            ),
        )


def _assemble_array(
    results: Sequence[CompositeTermResult],
    attribute: str,
) -> np.ndarray | None:
    values = [getattr(item, attribute) for item in results]
    if any(value is None for value in values):
        return None
    arrays = [np.asarray(value, dtype=float) for value in values]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"composite {attribute}s have inconsistent shapes")
    return sum(
        item.term.coefficient * array for item, array in zip(results, arrays, strict=True)
    )


__all__ = [
    "CompositeAssembly",
    "CompositeOrchestrator",
    "CompositeTerm",
    "CompositeTermResult",
]

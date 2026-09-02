"""PPCS2 energy model assembled from ORCA single-point calculations."""

from __future__ import annotations

from collections.abc import Mapping

from .scan import QMBackendTerm, QMScanBackend


PPCS2_ORCA_SCHEMA = "matrix.link.ppcs2_orca.v1"
PPCS2_ORCA_FORMULA = "F12-CCSD(T)/cc-pVDZ-F12 + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"


def ppcs2_orca_backend(
    *,
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    processors: int = 1,
    memory_gb: int | None = None,
    composite_parallel_workers: int = 1,
    tight_pno: bool = True,
    coupled_cluster_approximation: str = "canonical",
) -> QMScanBackend:
    """Return the all-ORCA PPCS2 composite surface used by LINK.

    The canonical explicitly correlated coupled-cluster contribution and both
    conventional MP2 core-valence terms are evaluated at the same Cartesian
    geometry.  The coupled-cluster term is energy-only, while the MP2 terms
    advertise their analytic gradients to the general composite orchestrator.
    """

    common = {
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "executable": executable,
        "timeout": timeout,
        "env": None if env is None else dict(env),
        "processors": int(processors),
        "memory_gb": memory_gb,
        "properties": ("energy",),
        "restart_reuse_for_displacements": False,
    }
    approximation = str(coupled_cluster_approximation).strip().lower()
    if approximation not in {"canonical", "dlpno"}:
        raise ValueError("coupled_cluster_approximation must be 'canonical' or 'dlpno'")
    pno_keyword = "TightPNO" if tight_pno else "NormalPNO"
    if approximation == "canonical":
        if int(multiplicity) > 1:
            # ORCA's canonical F12-CCSD(T) driver rejects UHF references;
            # the documented UHF-compatible route is the RI-F12 variant.
            high_method = "F12/RI-CCSD(T)"
            high_route = "UHF VeryTightSCF cc-pVDZ-F12-CABS cc-pVTZ/C"
            high_label = "F12/RI-CCSD(T)/2F12"
        else:
            high_method = "F12-CCSD(T)"
            high_route = "RHF VeryTightSCF cc-pVDZ-F12-CABS"
            high_label = "F12-CCSD(T)/2F12"
    else:
        high_method = "DLPNO-CCSD(T)-F12"
        high_route = f"RHF {pno_keyword} VeryTightSCF aug-cc-pVDZ/C cc-pVDZ-F12-CABS"
        high_label = "DLPNO-CCSD(T)-F12/2F12"
    coupled_cluster = QMScanBackend(
        name="orca",
        method=high_method,
        basis="cc-pVDZ-F12",
        route=high_route,
        gradient_mode="energy",
        **common,
    )
    all_electron_mp2 = QMScanBackend(
        name="orca",
        method="MP2",
        basis="cc-pwCVTZ",
        route="RHF NoFrozenCore VeryTightSCF",
        gradient_mode="analytic",
        **common,
    )
    frozen_core_mp2 = QMScanBackend(
        name="orca",
        method="MP2",
        basis="cc-pwCVTZ",
        route="RHF FrozenCore VeryTightSCF",
        gradient_mode="analytic",
        **common,
    )
    return QMScanBackend(
        name="linear_composite",
        charge=int(charge),
        multiplicity=int(multiplicity),
        gradient_mode="hybrid",
        properties=("energy",),
        composite_terms=(
            QMBackendTerm(high_label, +1.0, coupled_cluster),
            QMBackendTerm("ae-MP2/wC3", +1.0, all_electron_mp2),
            QMBackendTerm("fc-MP2/wC3", -1.0, frozen_core_mp2),
        ),
        composite_parallel_workers=int(composite_parallel_workers),
        resolution={
            "schema": PPCS2_ORCA_SCHEMA,
            "formula": PPCS2_ORCA_FORMULA,
            "implementation": "all_ORCA",
            "gradient_coordinates": "SONIC",
            "gradient_assembly": (
                "finite_difference_high_level_plus_analytic_MP2_core_valence"
            ),
            "term_derivative_sources": {
                high_label: "numerical_SONIC",
                "ae-MP2/wC3": "analytic_ORCA",
                "fc-MP2/wC3": "analytic_ORCA",
            },
            "coupled_cluster_approximation": approximation,
            "core_valence": "conventional_MP2_all_electron_minus_frozen_core",
        },
        restart_reuse_for_displacements=False,
    )


__all__ = ["PPCS2_ORCA_FORMULA", "PPCS2_ORCA_SCHEMA", "ppcs2_orca_backend"]

#!/usr/bin/env python3
"""Run the HCN/HNC PPCS2 surface from an imported GDV Hessian seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from matrix_chem import preprocess_to_enriched_xyz, write_validation_section
from matrix_link.optimizer import (
    OptimizerSettings,
    coordinate_model_from_xyzin,
    optimize_geometry,
    optimizer_hessian_from_engine_hessian,
)
from matrix_link.ppcs2 import ppcs2_orca_backend
from matrix_smith import write_gicforge_build_sections


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hessian-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dlpno", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    xyz = root / "input.xyz"
    xyz.write_text(
        "3\nHCN-HNC GDV B3LYP transition-state geometry\n"
        "H -1.049833 0.240728 0.000000\n"
        "C 0.080756 0.617003 0.000000\n"
        "N 0.080756 -0.563249 0.000000\n",
        encoding="utf-8",
    )
    xyzin = root / "input.xyzin"
    preprocess_to_enriched_xyz(xyz, xyzin)
    validation = write_validation_section(xyzin, require_fragments=False)
    if validation.status != "PASS":
        raise RuntimeError(f"topology validation failed: {validation.status}")
    write_gicforge_build_sections(xyzin, symmetrize=True)

    model = coordinate_model_from_xyzin(xyzin, kind="sonic")
    hessian = optimizer_hessian_from_engine_hessian(
        "gaussian", args.hessian_log, model
    )
    backend = ppcs2_orca_backend(
        charge=0,
        multiplicity=1,
        executable="/Users/vincenzobarone/orca_6_1_1/orca",
        processors=2,
        memory_gb=8,
        composite_parallel_workers=1,
        coupled_cluster_approximation="dlpno" if args.dlpno else "canonical",
    )
    result = optimize_geometry(
        xyzin,
        run_dir=root / "optimizer",
        coordinate_model=model,
        backend=backend,
        settings=OptimizerSettings(
            max_steps=20,
            fd_totally_symmetric_only=True,
            fd_stencil_policy="one_sided_only",
            fd_parallel_workers=3,
            stationary_point="transition_state",
            transition_mode=0,
            freeze_transition_reaction_mode=True,
        ),
        initial_hessian=hessian,
        initial_hessian_source=f"GDV B3LYP/def2TZVP Freq: {args.hessian_log}",
    )
    (root / "result.txt").write_text(
        f"status={result.status}\nconverged={result.converged}\n"
        f"iterations={len(result.iterations)}\n"
        f"energy_hartree={result.final_energy_hartree:.12f}\n",
        encoding="utf-8",
    )
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())

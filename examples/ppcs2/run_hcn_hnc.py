#!/usr/bin/env python3
"""Run a bounded HCN/HNC minimum and transition-state feasibility test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from matrix_link import read_optimization_input, run_optimization_input


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    if manifest.get("schema") != "matrix.link.ppcs2_hcn_hnc_feasibility.v1":
        raise ValueError("unsupported HCN/HNC feasibility manifest")

    minima = {}
    for name in ("hcn", "hnc"):
        result = run_optimization_input(source_dir / f"{name}.inp", run_dir=root / name)
        if not result.converged:
            raise RuntimeError(f"{name.upper()} B3LYP preparation did not converge")
        minima[name] = result

    # A perfectly linear midpoint is rank-deficient for internal-coordinate
    # construction.  This slightly bent planar H-transfer guess supplies the
    # full SONIC tangent while preserving the molecular plane; the C and N
    # atoms remain fixed in the common atom order H-C-N.  First locate the
    # saddle on B3LYP, then use that converged geometry for PPCS2.
    hcn = read_optimization_input(source_dir / "hcn.inp")
    b3lyp_ts_input = root / "hcn_hnc_b3lyp_ts.inp"
    b3lyp_ts_input.write_text(
        _ts_input_text(hcn, backend="orca"),
        encoding="utf-8",
    )
    b3lyp_ts = run_optimization_input(b3lyp_ts_input, run_dir=root / "b3lyp_ts")
    if not b3lyp_ts.converged:
        raise RuntimeError("B3LYP HCN/HNC transition-state search did not converge")

    ts_input = root / "hcn_hnc_ts.inp"
    ts_input.write_text(
        _ts_input_text(hcn, coordinates=b3lyp_ts.final_coordinates_angstrom),
        encoding="utf-8",
    )
    ts = run_optimization_input(ts_input, run_dir=root / "ts")
    summary = {
        "schema": "matrix.link.ppcs2_hcn_hnc_result.v1",
        "minima": {
            name: {
                "converged": bool(result.converged),
                "iterations": len(result.iterations),
                "energy_hartree": float(result.final_energy_hartree),
            }
            for name, result in minima.items()
        },
        "transition_state": {
            "b3lyp_converged": bool(b3lyp_ts.converged),
            "b3lyp_iterations": len(b3lyp_ts.iterations),
            "b3lyp_energy_hartree": float(b3lyp_ts.final_energy_hartree),
            "converged": bool(ts.converged),
            "status": ts.status,
            "iterations": len(ts.iterations),
            "energy_hartree": float(ts.final_energy_hartree),
            "summary": str(ts.summary_path),
        },
    }
    output = root / "hcn_hnc_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0 if ts.converged else 1


def _ts_input_text(source, *, coordinates=None, backend="ppcs2-orca") -> str:
    if coordinates is None:
        coordinates = np.asarray(
            [[0.59, 0.25, 0.0], [0.0, 0.0, 0.0], [1.18, 0.0, 0.0]],
            dtype=float,
        )
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (3, 3):
        raise ValueError("HCN/HNC transition-state coordinates must have shape (3, 3)")
    model_line = (
        "# MATRIX B3LYP/def2-TZVP Opt=SONIC D4 RIJCOSX def2/J TightSCF"
        if backend == "orca"
        else "# MATRIX Opt=SONIC"
    )
    lines = [
        f"%Backend={backend}",
        "%MaxSteps=20",
        "%Processors=1",
        "%MemoryGB=8",
        "%FDWorkers=3",
        "%StationaryPoint=transition_state",
        "%TransitionMode=0",
        model_line,
        "",
        "PPCS2 HCN to HNC transition state",
        "",
        f"{source.charge} {source.multiplicity}",
    ]
    lines.extend(
        f"{atom:2s} {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}"
        for atom, row in zip(("H", "C", "N"), coordinates, strict=True)
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

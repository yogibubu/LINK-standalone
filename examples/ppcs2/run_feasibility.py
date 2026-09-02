#!/usr/bin/env python3
"""Run the two bounded PPCS2 all-ORCA feasibility cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from matrix_link import (
    PointEvaluationResult,
    QMScanBackend,
    ScanPoint,
    read_optimization_input,
    run_optimization_input,
    run_qm_scan_points,
)
from matrix_chem.xyzin_geometry import read_xyzin_geometry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--only", type=str, default=None, help="run only this manifest case")
    parser.add_argument(
        "--skip-b3lyp",
        action="store_true",
        help="use the supplied geometry unchanged and only acquire a B3LYP Hessian",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "matrix.link.ppcs2_feasibility.v1":
        raise ValueError("unsupported PPCS2 feasibility manifest")
    cases = tuple(str(item) for item in payload.get("cases", ()))
    if args.only is not None:
        cases = tuple(item for item in cases if item == args.only)
    if not cases or len(cases) > 2:
        raise ValueError("the feasibility proof must contain one or two cases")

    root = args.run_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for name in cases:
        source = (manifest_path.parent / name).resolve()
        if source.parent != manifest_path.parent or not source.is_file():
            raise ValueError(f"invalid feasibility input: {name}")
        case_root = root / source.stem
        case_root.mkdir(parents=True, exist_ok=True)
        source_deck = read_optimization_input(source)
        supplied_geometry_is_high_quality = (
            "B2PLYP/3F12r" in source_deck.title or "LCB25" in source_deck.title
        )
        skip_b3lyp = args.skip_b3lyp or supplied_geometry_is_high_quality
        if skip_b3lyp:
            geometry_input = _write_source_xyzin(case_root / "supplied_geometry", source_deck)
            preparation = None
            final_coordinates = source_deck.coordinates_angstrom
        else:
            preparation_input = _write_input(
                case_root / "b3lyp_geometry.inp",
                source_deck,
                backend="orca",
                title=f"B3LYP preparation: {source_deck.title}",
                coordinates=source_deck.coordinates_angstrom,
                model_and_keywords="B3LYP/def2-TZVP Opt=Cartesian D4 RIJCOSX def2/J TightSCF",
                max_steps=30,
            )
            preparation = run_optimization_input(
                preparation_input,
                run_dir=case_root / "b3lyp_geometry",
            )
            if not preparation.converged:
                raise RuntimeError(f"B3LYP preparation did not converge for {source.stem}")
            geometry_input = case_root / "b3lyp_geometry" / "input.xyzin"
            final_coordinates = preparation.final_coordinates_angstrom
        hessian_path = _acquire_b3lyp_hessian(
            geometry_input,
            case_root / "b3lyp_hessian",
            source_deck,
        )
        ppcs2_input = _write_input(
            case_root / "ppcs2.inp",
            source_deck,
            backend="ppcs2-orca",
            title=f"PPCS2 from B3LYP geometry: {source_deck.title}",
            coordinates=final_coordinates,
            model_and_keywords="Opt=SONIC",
            max_steps=source_deck.max_steps,
            processors=1,
            initial_hessian=hessian_path,
        )
        result = run_optimization_input(ppcs2_input, run_dir=case_root / "ppcs2")
        records.append(
            {
                "case": source.stem,
                "preparation": {
                    "method": "ORCA B3LYP-D4/def2-TZVP",
                    "converged": preparation is not None,
                    "iterations": 0 if preparation is None else len(preparation.iterations),
                    "summary": None if preparation is None else str(preparation.summary_path),
                },
                "converged": bool(result.converged),
                "status": result.status,
                "iterations": len(result.iterations),
                "energy_hartree": float(result.final_energy_hartree),
                "qm_evaluations": int(result.qm_evaluations),
                "energy_evaluations": int(result.energy_evaluations),
                "summary": str(result.summary_path),
            }
        )
    summary = {
        "schema": "matrix.link.ppcs2_feasibility_result.v1",
        "manifest": str(manifest_path),
        "cases": records,
        "all_converged": all(bool(item["converged"]) for item in records),
    }
    summary_path = root / "feasibility_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    return 0 if summary["all_converged"] else 1


def _acquire_b3lyp_hessian(xyzin_path: Path, run_dir: Path, source_deck) -> Path:
    """Compute the Cartesian B3LYP Hessian at the converged B3LYP geometry."""

    geometry = read_xyzin_geometry(xyzin_path)
    point = ScanPoint(
        index=0,
        displacement=0.0,
        coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
    )
    backend = QMScanBackend(
        name="orca",
        method="B3LYP",
        basis="def2-TZVP",
        route="D4 RIJCOSX def2/J TightSCF Freq",
        charge=source_deck.charge,
        multiplicity=source_deck.multiplicity,
        executable=source_deck.executable,
        processors=source_deck.processors,
        memory_gb=source_deck.memory_gb,
        properties=("energy", "hessian"),
        gradient_mode="analytic",
    )
    result = run_qm_scan_points(
        xyzin_path,
        (point,),
        backend,
        run_dir=run_dir,
    )[0]
    if result.status != "completed" or result.hessian_hartree_per_bohr2 is None:
        raise RuntimeError(f"B3LYP Hessian failed: {result.message}")
    target = run_dir / "b3lyp_cartesian_hessian.json"
    target.write_text(
        json.dumps(
            {"cartesian_hessian": np.asarray(result.hessian_hartree_per_bohr2).tolist()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target.resolve()


def _write_source_xyzin(root: Path, source_deck) -> Path:
    """Create the standard enriched geometry input without changing coordinates."""

    from matrix_chem import preprocess_to_enriched_xyz

    root.mkdir(parents=True, exist_ok=True)
    xyz = root / "input.xyz"
    lines = [str(len(source_deck.atoms)), source_deck.title]
    lines.extend(
        f"{atom} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}"
        for atom, row in zip(source_deck.atoms, source_deck.coordinates_angstrom, strict=True)
    )
    xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")
    xyzin = root / "input.xyzin"
    preprocess_to_enriched_xyz(xyz, xyzin)
    return xyzin


def _write_input(
    path: Path,
    source,
    *,
    backend: str,
    title: str,
    coordinates,
    model_and_keywords: str,
    max_steps: int,
    processors: int | None = None,
    initial_hessian: Path | None = None,
) -> Path:
    if coordinates is None:
        raise ValueError("PPCS2 feasibility inputs require explicit Cartesian coordinates")
    lines = [
        f"%Backend={backend}",
        f"%MaxSteps={int(max_steps)}",
        f"%Processors={int(source.processors if processors is None else processors)}",
        f"%MemoryGB={int(source.memory_gb or 1)}",
        f"%FDWorkers={int(source.fd_parallel_workers)}",
        f"%CompositeWorkers={int(source.composite_parallel_workers)}",
    ]
    if initial_hessian is not None:
        lines.append(f"%InitialHessian={initial_hessian}")
    if source.executable:
        lines.append(f"%Executable={source.executable}")
    lines.extend(
        [
            f"# MATRIX {model_and_keywords}",
            "",
            title,
            "",
            f"{int(source.charge)} {int(source.multiplicity)}",
        ]
    )
    lines.extend(
        f"{atom:2s} {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}"
        for atom, row in zip(source.atoms, coordinates, strict=True)
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())

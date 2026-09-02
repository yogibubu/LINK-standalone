"""Combine two endpoint SONIC charts into one atlas-driven TS chart."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Sequence

import numpy as np

from matrix_chem import atomic_mass
from matrix_chem.topology.elements import atomic_number

from .contracts import GICForgeContractError
from .evaluation import build_gic_b_matrix, evaluate_gic_values
from .models import FrozenGIC, GICDefinition, GICPrimitive


ENDPOINT_SONIC_COMBINATION_SCHEMA = "matrix.smith.endpoint_sonic_combination.v1"


@dataclass(frozen=True)
class EndpointSonicCombination:
    """Auditable composition of common and endpoint-reactive SONICs."""

    schema: str
    definition: GICDefinition
    common_count: int
    start_reactive_gics: tuple[str, ...]
    end_reactive_gics: tuple[str, ...]
    source_labels: tuple[str, ...]
    source_irreps: tuple[str, ...]
    transformation: tuple[tuple[float, ...], ...]
    endpoint_displacement: tuple[float, ...]
    rank_start: int
    rank_end: int
    minimum_singular_value_start: float
    minimum_singular_value_end: float




def combine_endpoint_sonic_definitions(
    start: GICDefinition,
    end: GICDefinition,
    atom_symbols: Sequence[str],
    *,
    reaction_class_id: str,
    rank_tolerance: float = 1.0e-9,
) -> EndpointSonicCombination:
    """Build one unsymmetrized exact-rank chart from two endpoint SONIC sets.

    Coordinates whose normalized primitive definitions are identical are kept
    unchanged. The remaining endpoint coordinates are pooled, projected out
    of the common block in mass-weighted Cartesian space, and reduced to the
    missing vibrational rank by one deterministic joint endpoint SVD. The
    resulting linear combinations are accepted only when the complete chart
    has full rank at both minima and at their aligned Cartesian midpoint.

    This is the general exploration construction. It deliberately carries no
    parent-point-group constraint; symmetry is perceived again from the
    selected Cartesian seed when the standard TS exploitation phase starts.
    """

    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank tolerance must be finite and positive")
    reaction_class = str(reaction_class_id).strip().upper()
    if not reaction_class:
        raise ValueError("reaction class identifier cannot be empty")
    if start.symmetrize or end.symmetrize:
        raise GICForgeContractError(
            "general endpoint SONIC composition requires unsymmetrized exploration charts"
        )
    start_xyz = np.asarray(start.reference_coordinates_angstrom, dtype=float)
    end_xyz = np.asarray(end.reference_coordinates_angstrom, dtype=float)
    if start_xyz.shape != end_xyz.shape or start_xyz.ndim != 2 or start_xyz.shape[1] != 3:
        raise GICForgeContractError("endpoint SONIC geometries must have identical shape")
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    if len(symbols) != start_xyz.shape[0]:
        raise GICForgeContractError("atom symbols do not match endpoint SONIC geometries")
    if (
        start.rank != start.target_rank
        or end.rank != end.target_rank
        or start.target_rank != end.target_rank
    ):
        raise GICForgeContractError(
            "endpoint SONIC charts must have the same complete vibrational rank"
        )
    target_rank = int(start.target_rank)

    start_signatures = _exploration_gics_by_signature(start)
    end_signatures = _exploration_gics_by_signature(end)
    common_signatures = start_signatures.keys() & end_signatures.keys()
    common_start = tuple(
        gic
        for gic in start.gics
        if _exploration_gic_signature(start, gic) in common_signatures
    )
    start_reactive = tuple(
        gic
        for gic in start.gics
        if _exploration_gic_signature(start, gic) not in common_signatures
    )
    end_reactive = tuple(
        gic
        for gic in end.gics
        if _exploration_gic_signature(end, gic) not in common_signatures
    )
    common_count = len(common_start)
    reactive_rank = target_rank - common_count
    if reactive_rank <= 0:
        raise GICForgeContractError(
            "endpoint topology change has no endpoint-specific SONIC rank"
        )

    selected = tuple((start, gic) for gic in common_start + start_reactive) + tuple(
        (end, gic) for gic in end_reactive
    )
    primitives, primitive_id_by_key = _unified_exploration_primitives(selected)
    combination_backend = (
        "oracle-native-primitive.v1"
        if any(
            primitive.function in {"FROT", "FTRANS"} for primitive in primitives
        )
        else start.backend
    )
    common = tuple(
        replace(
            _remap_exploration_gic(start, gic, primitive_id_by_key),
            irrep="A",
        )
        for gic in common_start
    )
    source_gics = tuple(
        replace(
            _remap_exploration_gic(start, gic, primitive_id_by_key),
            irrep="A",
        )
        for gic in start_reactive
    ) + tuple(
        replace(
            _remap_exploration_gic(end, gic, primitive_id_by_key),
            irrep="A",
        )
        for gic in end_reactive
    )
    oriented_primitives, oriented_gics = _oriented_reaction_domain_coordinates(
        start,
        end,
        start_xyz,
        end_xyz,
        existing_primitives=primitives,
        tolerance=tolerance,
    )
    if oriented_primitives:
        primitives = (*primitives, *oriented_primitives)
        source_gics = (*source_gics, *oriented_gics)
    if len(source_gics) < reactive_rank:
        raise GICForgeContractError(
            "endpoint-specific SONIC pool is smaller than the missing vibrational rank"
        )
    source_definition = replace(
        start,
        backend=combination_backend,
        point_group="C1",
        symmetrize=False,
        symmetry_diagnostics=None,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in start_xyz
        ),
        primitives=primitives,
        gics=common + source_gics,
        candidate_count=len(primitives),
        periodic_coordinate_estimates=(),
        primitive_b_matrix_sha256="",
    )
    rows_start = np.asarray(
        build_gic_b_matrix(source_definition, coordinates_angstrom=start_xyz).rows,
        dtype=float,
    )
    rows_end = np.asarray(
        build_gic_b_matrix(source_definition, coordinates_angstrom=end_xyz).rows,
        dtype=float,
    )
    midpoint_xyz = 0.5 * (start_xyz + end_xyz)
    rows_midpoint = np.asarray(
        build_gic_b_matrix(
            source_definition, coordinates_angstrom=midpoint_xyz
        ).rows,
        dtype=float,
    )
    values_start = evaluate_gic_values(
        source_definition, coordinates_angstrom=start_xyz
    )
    values_end = evaluate_gic_values(source_definition, coordinates_angstrom=end_xyz)
    masses = np.asarray([atomic_mass(_required_atomic_number(item)) for item in symbols])
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    weighted_start = rows_start * inverse_sqrt_mass[None, :]
    weighted_midpoint = rows_midpoint * inverse_sqrt_mass[None, :]
    weighted_end = rows_end * inverse_sqrt_mass[None, :]
    oriented_absolute_indices = tuple(
        index
        for index, gic in enumerate(source_definition.gics)
        if gic.family == "OUT_OF_PLANE"
        and gic.name.startswith("REACTION_ORIENTED_U(")
    )
    retained_common = _joint_independent_row_indices(
        (
            weighted_start[:common_count],
            weighted_midpoint[:common_count],
            weighted_end[:common_count],
        ),
        tolerance,
        fixed_matrices=(
            weighted_start[np.asarray(oriented_absolute_indices, dtype=int)],
            weighted_midpoint[np.asarray(oriented_absolute_indices, dtype=int)],
            weighted_end[np.asarray(oriented_absolute_indices, dtype=int)],
        ),
    )
    retained_set = set(retained_common)
    demoted_common = tuple(
        index for index in range(common_count) if index not in retained_set
    )
    source_indices = (*demoted_common, *range(common_count, len(source_definition.gics)))
    order = (*retained_common, *source_indices)
    all_source_order_gics = common + source_gics
    common = tuple(common[index] for index in retained_common)
    source_gics = tuple(
        all_source_order_gics[index] for index in source_indices
    )
    common_count = len(common)
    reactive_rank = target_rank - common_count
    ordered_start = weighted_start[np.asarray(order, dtype=int)]
    ordered_midpoint = weighted_midpoint[np.asarray(order, dtype=int)]
    ordered_end = weighted_end[np.asarray(order, dtype=int)]
    ordered_values_start = values_start[np.asarray(order, dtype=int)]
    ordered_values_end = values_end[np.asarray(order, dtype=int)]
    common_start_rows = ordered_start[:common_count]
    common_midpoint_rows = ordered_midpoint[:common_count]
    common_end_rows = ordered_end[:common_count]
    source_start_rows = ordered_start[common_count:]
    source_midpoint_rows = ordered_midpoint[common_count:]
    source_end_rows = ordered_end[common_count:]
    source_delta = (
        ordered_values_end[common_count:] - ordered_values_start[common_count:]
    )
    oriented_indices = tuple(
        index
        for index, gic in enumerate(source_gics)
        if gic.family == "OUT_OF_PLANE"
        and gic.name.startswith("REACTION_ORIENTED_U(")
    )
    if len(oriented_indices) > reactive_rank:
        raise GICForgeContractError(
            "oriented reaction coordinates exceed the missing vibrational rank"
        )
    oriented_set = set(oriented_indices)
    reduced_indices = tuple(
        index for index in range(len(source_gics)) if index not in oriented_set
    )
    reduced_rank = reactive_rank - len(oriented_indices)

    fixed_start_rows = source_start_rows[np.asarray(oriented_indices, dtype=int)]
    fixed_midpoint_rows = source_midpoint_rows[
        np.asarray(oriented_indices, dtype=int)
    ]
    fixed_end_rows = source_end_rows[np.asarray(oriented_indices, dtype=int)]
    extended_common_start = np.vstack((common_start_rows, fixed_start_rows))
    extended_common_midpoint = np.vstack(
        (common_midpoint_rows, fixed_midpoint_rows)
    )
    extended_common_end = np.vstack((common_end_rows, fixed_end_rows))
    for label, rows in (
        ("start", extended_common_start),
        ("midpoint", extended_common_midpoint),
        ("end", extended_common_end),
    ):
        if int(np.linalg.matrix_rank(rows, tol=tolerance)) != len(rows):
            raise GICForgeContractError(
                f"oriented reaction coordinate is dependent at the {label} geometry"
            )

    reduced_start_rows = source_start_rows[np.asarray(reduced_indices, dtype=int)]
    reduced_midpoint_rows = source_midpoint_rows[
        np.asarray(reduced_indices, dtype=int)
    ]
    reduced_end_rows = source_end_rows[np.asarray(reduced_indices, dtype=int)]
    projected_start = reduced_start_rows @ _row_complement_projector(
        extended_common_start, tolerance
    )
    projected_end = reduced_end_rows @ _row_complement_projector(
        extended_common_end, tolerance
    )
    projected_midpoint = reduced_midpoint_rows @ _row_complement_projector(
        extended_common_midpoint, tolerance
    )
    if (
        int(np.linalg.matrix_rank(projected_start, tol=tolerance)) < reduced_rank
        or int(np.linalg.matrix_rank(projected_end, tol=tolerance)) < reduced_rank
        or int(np.linalg.matrix_rank(projected_midpoint, tol=tolerance))
        < reduced_rank
    ):
        raise GICForgeContractError(
            "endpoint-specific SONIC pool cannot restore full rank at endpoints and midpoint"
        )
    fixed_transformation = np.zeros(
        (len(oriented_indices), len(source_gics)), dtype=float
    )
    for row, source_index in enumerate(oriented_indices):
        fixed_transformation[row, source_index] = 1.0
    reduced_transformation = np.zeros(
        (reduced_rank, len(source_gics)), dtype=float
    )
    if reduced_rank:
        norms = np.sqrt(
            (1.0 / 3.0)
            * (
                np.sum(projected_start * projected_start, axis=1)
                + np.sum(projected_end * projected_end, axis=1)
                + np.sum(projected_midpoint * projected_midpoint, axis=1)
            )
        )
        valid = np.isfinite(norms) & (norms > tolerance)
        if int(np.count_nonzero(valid)) < reduced_rank:
            raise GICForgeContractError(
                "endpoint-specific SONIC is singular after common-block projection"
            )
        valid_indices = tuple(
            source_index
            for source_index, keep in zip(reduced_indices, valid, strict=True)
            if bool(keep)
        )
        norms = norms[valid]
        projected_start = projected_start[valid]
        projected_end = projected_end[valid]
        projected_midpoint = projected_midpoint[valid]
        normalized_start = projected_start / norms[:, None]
        normalized_end = projected_end / norms[:, None]
        normalized_midpoint = projected_midpoint / norms[:, None]
        pooled = np.hstack((normalized_start, normalized_midpoint, normalized_end))
        left, _singular, _right = np.linalg.svd(pooled, full_matrices=False)
        reduced_local = left[:, :reduced_rank].T / norms[None, :]
        reduced_transformation[:, np.asarray(valid_indices, dtype=int)] = reduced_local
        reduced_transformation = _reaction_aligned_rows(
            reduced_transformation,
            reduced_transformation @ source_delta,
            tolerance,
        )
    transformation = np.vstack((fixed_transformation, reduced_transformation))

    source_coefficients = _gic_coefficient_matrix(source_gics, primitives)
    combined_coefficients = transformation @ source_coefficients
    for row in range(combined_coefficients.shape[0]):
        norm = float(np.linalg.norm(combined_coefficients[row]))
        if not np.isfinite(norm) or norm <= tolerance:
            raise GICForgeContractError("combined reactive SONIC has zero coefficients")
        combined_coefficients[row] /= norm
        transformation[row] /= norm
    reactive_gics = tuple(
        _combined_gic(index, "A", coefficients, primitives)
        for index, coefficients in enumerate(combined_coefficients, 1)
    )
    final_gics = tuple(
        _renumber_gic(gic, index)
        for index, gic in enumerate(common + reactive_gics, 1)
    )
    combined_definition = replace(
        start,
        backend=combination_backend,
        point_group="C1",
        symmetrize=False,
        symmetry_diagnostics=None,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in start_xyz
        ),
        primitives=primitives,
        gics=final_gics,
        target_rank=target_rank,
        rank=target_rank,
        candidate_count=len(primitives),
        reduction_diagnostics=None,
        periodic_coordinate_estimates=(),
        semantic_diagnostics=(
            *start.semantic_diagnostics,
            "ENDPOINT_SONIC_COMBINATION "
            f"CLASS={reaction_class} COMMON={common_count} "
            f"REACTIVE={len(source_gics)}->{reactive_rank} "
            f"ORIENTED_REACTION_COORDINATES={len(oriented_gics)} "
            "SYMMETRY=EXPLORATION_DYNAMIC",
        ),
        primitive_source="ENDPOINT_SONIC_COMBINATION",
        primitive_source_schema=ENDPOINT_SONIC_COMBINATION_SCHEMA,
        primitive_b_matrix_sha256="",
        wilson_tangent_rank=0,
        wilson_tangent_singular_min=0.0,
        wilson_tangent_singular_max=0.0,
    )
    endpoint_singular: list[np.ndarray] = []
    for label, coordinates in (
        ("start", start_xyz),
        ("midpoint", midpoint_xyz),
        ("end", end_xyz),
    ):
        rows = np.asarray(
            build_gic_b_matrix(
                combined_definition, coordinates_angstrom=coordinates
            ).rows,
            dtype=float,
        ) * inverse_sqrt_mass[None, :]
        singular = np.linalg.svd(rows, compute_uv=False)
        if int(np.linalg.matrix_rank(rows, tol=tolerance)) != target_rank:
            raise GICForgeContractError(
                f"combined endpoint SONIC chart loses rank at the {label} geometry"
            )
        endpoint_singular.append(singular)
    midpoint_singular = endpoint_singular[1]
    combined_definition = replace(
        combined_definition,
        wilson_tangent_rank=target_rank,
        wilson_tangent_singular_min=float(midpoint_singular[-1]),
        wilson_tangent_singular_max=float(midpoint_singular[0]),
    )
    displacement = transformation @ source_delta
    return EndpointSonicCombination(
        schema=ENDPOINT_SONIC_COMBINATION_SCHEMA,
        definition=combined_definition,
        common_count=common_count,
        start_reactive_gics=tuple(gic.name for gic in start_reactive),
        end_reactive_gics=tuple(gic.name for gic in end_reactive),
        source_labels=tuple(gic.name for gic in source_gics),
        source_irreps=tuple("A" for _gic in source_gics),
        transformation=tuple(
            tuple(float(value) for value in row) for row in transformation
        ),
        endpoint_displacement=tuple(float(value) for value in displacement),
        rank_start=target_rank,
        rank_end=target_rank,
        minimum_singular_value_start=float(endpoint_singular[0][-1]),
        minimum_singular_value_end=float(endpoint_singular[2][-1]),
    )


def _oriented_reaction_domain_coordinates(
    start: GICDefinition,
    end: GICDefinition,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    existing_primitives: Sequence[GICPrimitive],
    tolerance: float,
) -> tuple[tuple[GICPrimitive, ...], tuple[FrozenGIC, ...]]:
    """Add signed local-volume coordinates when an attachment changes.

    Distances and valence angles do not distinguish the two Cartesian sheets
    of a pyramidal inversion.  At a center that keeps at least three bonded
    neighbours while exchanging another attachment, one oriented
    out-of-plane coordinate supplies the missing sheet label.  It enters the
    endpoint-only source pool and is subsequently reduced with all other
    reactive sources; the final chart therefore remains exactly 3N-6.
    A candidate is mandatory only when its endpoint values have opposite
    signs; a merely different pyramidalization belongs to the ordinary
    endpoint source reduction and must not be mistaken for an inversion.
    """

    start_adjacency = _bond_adjacency_from_definition(start)
    end_adjacency = _bond_adjacency_from_definition(end)
    used_keys = {_exploration_primitive_key(item) for item in existing_primitives}
    primitives: list[GICPrimitive] = []
    gics: list[FrozenGIC] = []
    for center in sorted(start_adjacency.keys() | end_adjacency.keys()):
        start_neighbours = start_adjacency.get(center, set())
        end_neighbours = end_adjacency.get(center, set())
        if start_neighbours == end_neighbours:
            continue
        common_neighbours = tuple(sorted(start_neighbours & end_neighbours))
        if len(common_neighbours) < 3:
            continue
        for plane1, plane2, out in combinations(common_neighbours, 3):
            serial = len(primitives) + 1
            primitive = GICPrimitive(
                identifier=f"RXU{serial:03d}",
                name=f"RXU{serial:03d}",
                family="OUT_OF_PLANE",
                function="U",
                atoms=(center, plane1, plane2, out),
                provenance="ENDPOINT_ATTACHMENT_CHANGE_ORIENTED_VOLUME",
                semantic_id=f"REACTION_ORIENTED_U:{center}:{plane1}:{plane2}:{out}",
                semantic_type="REACTION_ORIENTED_OUT_OF_PLANE",
            )
            if _exploration_primitive_key(primitive) in used_keys:
                continue
            probe_gic = FrozenGIC(
                identifier=f"RXQ{serial:03d}",
                name=f"REACTION_ORIENTED_U({center},{plane1},{plane2},{out})",
                family="OUT_OF_PLANE",
                irrep="A",
                primitive_id=primitive.identifier,
                gaussian_expression=primitive.gaussian_expression(),
            )
            probe = replace(
                start,
                backend="oracle-native-primitive.v1",
                point_group="C1",
                symmetrize=False,
                primitives=(primitive,),
                gics=(probe_gic,),
                target_rank=1,
                rank=1,
                reference_coordinates_angstrom=tuple(
                    tuple(float(value) for value in row) for row in start_xyz
                ),
                periodic_coordinate_estimates=(),
                primitive_b_matrix_sha256="",
            )
            try:
                start_value = float(
                    evaluate_gic_values(
                        probe, coordinates_angstrom=start_xyz
                    )[0]
                )
                end_value = float(
                    evaluate_gic_values(
                        probe, coordinates_angstrom=end_xyz
                    )[0]
                )
                rows = tuple(
                    np.asarray(
                        build_gic_b_matrix(probe, coordinates_angstrom=coordinates).rows,
                        dtype=float,
                    )[0]
                    for coordinates in (start_xyz, 0.5 * (start_xyz + end_xyz), end_xyz)
                )
            except (ArithmeticError, FloatingPointError, ValueError):
                continue
            orientation_threshold = max(1.0e-3, 100.0 * tolerance)
            if (
                min(abs(start_value), abs(end_value)) <= orientation_threshold
                or start_value * end_value >= 0.0
                or any(
                    np.any(~np.isfinite(row))
                    or float(np.linalg.norm(row)) <= tolerance
                    for row in rows
                )
            ):
                continue
            used_keys.add(_exploration_primitive_key(primitive))
            primitives.append(primitive)
            gics.append(probe_gic)
            break
    return tuple(primitives), tuple(gics)


def _bond_adjacency_from_definition(
    definition: GICDefinition,
) -> dict[int, set[int]]:
    """Recover only the explicit covalent stretches frozen in one SONIC chart."""

    adjacency: dict[int, set[int]] = {}
    for primitive in definition.primitives:
        if primitive.function != "R" or len(primitive.atoms) != 2:
            continue
        left, right = (int(atom) for atom in primitive.atoms)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def _exploration_primitive_key(primitive: GICPrimitive) -> tuple[object, ...]:
    atoms = tuple(int(atom) for atom in primitive.atoms)
    if primitive.function == "R":
        atoms = tuple(sorted(atoms))
    elif primitive.function in {"A", "D"}:
        atoms = min(atoms, tuple(reversed(atoms)))
    return (
        primitive.function,
        atoms,
        int(primitive.mode),
        tuple(int(atom) for atom in primitive.ref_atoms),
        tuple(primitive.refs),
        tuple(int(atom) for atom in primitive.frame_atoms),
        tuple(int(atom) for atom in primitive.ref_frame_atoms),
    )


def _normalized_exploration_gic_terms(
    definition: GICDefinition,
    gic: FrozenGIC,
) -> tuple[tuple[tuple[object, ...], float], ...]:
    primitive_by_id = {
        primitive.identifier: primitive for primitive in definition.primitives
    }
    terms = sorted(
        (
            _exploration_primitive_key(primitive_by_id[primitive_id]),
            float(coefficient),
        )
        for primitive_id, coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        )
    )
    norm = float(np.linalg.norm([coefficient for _key, coefficient in terms]))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise GICForgeContractError(f"SONIC {gic.identifier} has zero coefficients")
    normalized = [(key, coefficient / norm) for key, coefficient in terms]
    if normalized and normalized[0][1] < 0.0:
        normalized = [(key, -coefficient) for key, coefficient in normalized]
    return tuple((key, round(coefficient, 11)) for key, coefficient in normalized)


def _exploration_gic_signature(
    definition: GICDefinition,
    gic: FrozenGIC,
) -> tuple[object, ...]:
    return gic.family, _normalized_exploration_gic_terms(definition, gic)


def _exploration_gics_by_signature(
    definition: GICDefinition,
) -> dict[tuple[object, ...], FrozenGIC]:
    result: dict[tuple[object, ...], FrozenGIC] = {}
    for gic in definition.gics:
        signature = _exploration_gic_signature(definition, gic)
        if signature in result:
            raise GICForgeContractError(
                "endpoint SONIC definition contains duplicate coordinates"
            )
        result[signature] = gic
    return result


def _unified_exploration_primitives(
    selected_gics: Sequence[tuple[GICDefinition, FrozenGIC]],
) -> tuple[tuple[GICPrimitive, ...], dict[tuple[object, ...], str]]:
    primitive_by_key: dict[tuple[object, ...], GICPrimitive] = {}
    for definition, gic in selected_gics:
        by_id = {
            primitive.identifier: primitive for primitive in definition.primitives
        }
        for primitive_id, _coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        ):
            primitive = by_id[primitive_id]
            primitive_by_key.setdefault(
                _exploration_primitive_key(primitive), primitive
            )
    ordered = tuple(primitive_by_key)
    identifier_by_key = {
        key: f"P{index:03d}" for index, key in enumerate(ordered, 1)
    }
    primitives = tuple(
        replace(
            primitive_by_key[key],
            identifier=identifier_by_key[key],
            chart="PRINCIPAL",
            chart_reference_radian=None,
        )
        for key in ordered
    )
    return primitives, identifier_by_key


def _remap_exploration_gic(
    definition: GICDefinition,
    gic: FrozenGIC,
    identifier_by_key: dict[tuple[object, ...], str],
) -> FrozenGIC:
    primitive_by_id = {
        primitive.identifier: primitive for primitive in definition.primitives
    }
    coefficients = tuple(
        (
            identifier_by_key[
                _exploration_primitive_key(primitive_by_id[primitive_id])
            ],
            float(coefficient),
        )
        for primitive_id, coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        )
    )
    return replace(
        gic,
        primitive_id=coefficients[0][0],
        gaussian_expression="LINEAR_COMBINATION",
        coefficients=coefficients,
    )


def _required_atomic_number(symbol: str) -> int:
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise GICForgeContractError(f"unsupported atom symbol in SONIC composition: {symbol}")
    return int(number)




def _gic_coefficient_matrix(
    gics: Sequence[FrozenGIC],
    primitives: Sequence[GICPrimitive],
) -> np.ndarray:
    index = {primitive.identifier: column for column, primitive in enumerate(primitives)}
    result = np.zeros((len(gics), len(primitives)), dtype=float)
    for row, gic in enumerate(gics):
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            result[row, index[primitive_id]] += float(coefficient)
    return result


def _combined_gic(
    index: int,
    irrep: str,
    coefficients: np.ndarray,
    primitives: Sequence[GICPrimitive],
) -> FrozenGIC:
    terms = tuple(
        (primitive.identifier, float(coefficient))
        for primitive, coefficient in zip(primitives, coefficients, strict=True)
        if abs(float(coefficient)) > 1.0e-12
    )
    return FrozenGIC(
        identifier=f"RXN{index:03d}",
        name=f"Reaction{index:03d}",
        family="REACTION_COORDINATE",
        irrep=irrep,
        primitive_id=terms[0][0],
        gaussian_expression="LINEAR_COMBINATION",
        coefficients=terms,
    )


def _renumber_gic(gic: FrozenGIC, index: int) -> FrozenGIC:
    return replace(gic, identifier=f"GIC{index:03d}")


def _row_complement_projector(rows: np.ndarray, tolerance: float) -> np.ndarray:
    columns = rows.shape[1]
    if rows.size == 0:
        return np.eye(columns, dtype=float)
    _u_matrix, singular, vt_matrix = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.count_nonzero(singular > tolerance))
    basis = vt_matrix[:rank]
    return np.eye(columns, dtype=float) - basis.T @ basis


def _joint_independent_row_indices(
    matrices: Sequence[np.ndarray],
    tolerance: float,
    *,
    fixed_matrices: Sequence[np.ndarray] | None = None,
) -> tuple[int, ...]:
    """Select the largest deterministic row block independent in every matrix.

    ``fixed_matrices`` contains mandatory rows that seed each row space.  This
    is used for signed reaction-domain coordinates: a formally common SONIC
    row is demoted when retaining it would make an oriented coordinate
    dependent at any protected geometry.
    """

    values = tuple(np.asarray(matrix, dtype=float) for matrix in matrices)
    if not values:
        return ()
    row_count = values[0].shape[0]
    if any(matrix.ndim != 2 or matrix.shape[0] != row_count for matrix in values):
        raise ValueError("joint row-selection matrices have incompatible shapes")
    if any(not np.all(np.isfinite(matrix)) for matrix in values):
        raise ValueError("joint row-selection matrices contain non-finite values")
    if row_count == 0:
        return ()
    scales = np.sqrt(
        sum(np.sum(matrix * matrix, axis=1) for matrix in values) / len(values)
    )
    available = {
        index for index, scale in enumerate(scales) if float(scale) > tolerance
    }
    normalized = tuple(matrix / scales[:, None] for matrix in values)
    bases: list[list[np.ndarray]] = [[] for _matrix in normalized]
    if fixed_matrices is not None:
        fixed = tuple(np.asarray(matrix, dtype=float) for matrix in fixed_matrices)
        if len(fixed) != len(values) or any(
            matrix.ndim != 2 or matrix.shape[1] != values[index].shape[1]
            for index, matrix in enumerate(fixed)
        ):
            raise ValueError("fixed joint row-selection matrices have incompatible shapes")
        if any(not np.all(np.isfinite(matrix)) for matrix in fixed):
            raise ValueError("fixed joint row-selection matrices contain non-finite values")
        fixed_count = fixed[0].shape[0]
        if any(matrix.shape[0] != fixed_count for matrix in fixed):
            raise ValueError("fixed joint row-selection matrices have different row counts")
        for row_index in range(fixed_count):
            joint_scale = float(
                np.sqrt(
                    sum(float(matrix[row_index] @ matrix[row_index]) for matrix in fixed)
                    / len(fixed)
                )
            )
            if not np.isfinite(joint_scale) or joint_scale <= tolerance:
                raise GICForgeContractError(
                    "mandatory reaction coordinate has a zero Wilson row"
                )
            residuals = tuple(
                _twice_orthogonalized_vector(
                    matrix[row_index] / joint_scale,
                    basis,
                )
                for matrix, basis in zip(fixed, bases, strict=True)
            )
            if min(float(np.linalg.norm(residual)) for residual in residuals) <= tolerance:
                raise GICForgeContractError(
                    "mandatory reaction coordinates are jointly dependent"
                )
            for basis, residual in zip(bases, residuals, strict=True):
                basis.append(residual / np.linalg.norm(residual))
    selected: list[int] = []
    while available:
        scored: list[tuple[float, int, tuple[np.ndarray, ...]]] = []
        for index in sorted(available):
            residuals = tuple(
                _twice_orthogonalized_vector(matrix[index], basis)
                for matrix, basis in zip(normalized, bases, strict=True)
            )
            score = min(float(np.linalg.norm(residual)) for residual in residuals)
            scored.append((score, index, residuals))
        maximum = max(score for score, _index, _residuals in scored)
        if maximum <= tolerance:
            break
        score, pivot, residuals = next(
            item for item in scored if item[0] >= maximum - 1.0e-12
        )
        if score <= tolerance:
            break
        for basis, residual in zip(bases, residuals, strict=True):
            basis.append(residual / np.linalg.norm(residual))
        selected.append(pivot)
        available.remove(pivot)
    return tuple(selected)


def _twice_orthogonalized_vector(
    vector: np.ndarray,
    basis: Sequence[np.ndarray],
) -> np.ndarray:
    residual = np.array(vector, dtype=float, copy=True)
    if not basis:
        return residual
    orthonormal = np.vstack(basis)
    for _pass in range(2):
        residual -= (orthonormal @ residual) @ orthonormal
    return residual




def _reaction_aligned_rows(
    transformation: np.ndarray,
    displacement: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    norm = float(np.linalg.norm(displacement))
    if norm <= tolerance:
        return _deterministic_row_signs(transformation)
    first = displacement / norm
    _u_matrix, _singular, vt_matrix = np.linalg.svd(first.reshape(1, -1))
    aligned = np.vstack((first, vt_matrix[1:])) @ transformation
    aligned = _deterministic_row_signs(aligned)
    if float(aligned[0] @ transformation.T @ displacement) < 0.0:
        aligned[0] *= -1.0
    return aligned


def _deterministic_row_signs(rows: np.ndarray) -> np.ndarray:
    result = np.array(rows, dtype=float, copy=True)
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return result


__all__ = [
    "ENDPOINT_SONIC_COMBINATION_SCHEMA",
    "EndpointSonicCombination",
    "combine_endpoint_sonic_definitions",
]

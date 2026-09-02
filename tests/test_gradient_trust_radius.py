import math

import numpy as np
import pytest

from matrix_link.optimizer import OptimizerSettings, _effective_gradient_trust_radius


def test_gradient_adapts_minimum_trust_radius_far_from_convergence():
    settings = OptimizerSettings(trust_radius=0.3, max_force_tolerance=1.0e-3)
    force_max = 0.1
    expected = 0.3 * math.sqrt(1.0e-3 / force_max)
    actual = _effective_gradient_trust_radius(0.25, np.array([force_max, 0.0]), settings)
    assert actual == pytest.approx(expected)


def test_gradient_adaptation_is_disabled_near_convergence_and_for_ts():
    minimum = OptimizerSettings(trust_radius=0.3, max_force_tolerance=1.0e-3)
    assert _effective_gradient_trust_radius(
        0.25, np.array([9.0e-3, 0.0]), minimum
    ) == pytest.approx(0.25)

    transition_state = OptimizerSettings(
        trust_radius=0.3,
        max_force_tolerance=1.0e-3,
        stationary_point="transition_state",
    )
    assert _effective_gradient_trust_radius(
        0.25, np.array([0.1, 0.0]), transition_state
    ) == pytest.approx(0.25)


def test_adaptation_never_increases_controller_radius():
    settings = OptimizerSettings(trust_radius=0.3, max_force_tolerance=1.0e-3)
    assert _effective_gradient_trust_radius(0.1, np.array([0.01]), settings) == pytest.approx(0.1)

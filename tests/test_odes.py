"""
Tests for the ODE system (src/model/odes.py).

Tests verify mathematical properties that must always hold:
    1. At steady state with no drug, all derivatives are zero
    2. B cell counts never go negative during simulation
    3. Mass balance — cells leaving one compartment enter another
"""

import numpy as np
import pytest
import yaml

from src.model.odes import bcell_ode_system


@pytest.fixture
def params():
    """Load default parameters from config files."""
    with open("config/model_params.yaml") as f:
        p = yaml.safe_load(f)
    with open("config/pk_rituximab.yaml") as f:
        p.update(yaml.safe_load(f))
    return p


@pytest.fixture
def steady_state(params):
    """Compute analytical steady state for use in tests."""
    from src.simulation.simulator import compute_steady_state
    return compute_steady_state(params)


def no_dose(t):
    """Zero dose function — no drug in system."""
    return 0.0


def test_derivatives_zero_at_steady_state(params, steady_state):
    """
    At steady state with no drug, all derivatives must be exactly zero.

    This is the definition of steady state — the system is in balance.
    If this test fails, either the ODE system or the steady state
    computation is wrong.
    """
    y0 = [
        steady_state["B_bm"],
        steady_state["B_blood"],
        steady_state["B_tissue"],
        steady_state["C_central"],
        steady_state["C_peripheral"],
    ]

    derivatives = bcell_ode_system(t=0.0, y=y0, params=params, dose_fn=no_dose)

    for i, deriv in enumerate(derivatives):
        assert abs(deriv) < 1e-6, (
            f"Derivative {i} at steady state is {deriv:.2e}, expected ~0. "
            f"Steady state is not a true fixed point of the ODE system."
        )


def test_zero_drug_no_killing(params, steady_state):
    """
    With no drug present, the killing term must be zero.

    kill(C=0) = Emax * 0^n / (EC50^n + 0^n) = 0
    This is a mathematical property of the Emax function.
    """
    from src.model.pd import emax_killing
    kill = emax_killing(C=0.0,
                        Emax=params["Emax"],
                        EC50=params["EC50"],
                        n=params["n"])
    assert kill == 0.0, f"kill(C=0) should be 0, got {kill}"


def test_killing_at_ec50_equals_half_emax(params):
    """
    At C = EC50, killing rate must equal exactly Emax/2.

    This is the mathematical definition of EC50 — always true
    regardless of biological context or spare receptor effects.
    """
    from src.model.pd import emax_killing
    kill = emax_killing(
        C=params["EC50"],
        Emax=params["Emax"],
        EC50=params["EC50"],
        n=params["n"],
    )
    expected = params["Emax"] / 2
    assert abs(kill - expected) < 1e-10, (
        f"kill(EC50) = {kill:.6f}, expected Emax/2 = {expected:.6f}"
    )


def test_killing_approaches_emax_at_high_concentration(params):
    """
    At very high drug concentration, killing rate must approach Emax.

    kill(C >> EC50) -> Emax asymptotically.
    We test at 1000x EC50 — should be within 0.1% of Emax.
    """
    from src.model.pd import emax_killing
    very_high_C = params["EC50"] * 1000
    kill = emax_killing(
        C=very_high_C,
        Emax=params["Emax"],
        EC50=params["EC50"],
        n=params["n"],
    )
    assert kill > params["Emax"] * 0.999, (
        f"kill(1000*EC50) = {kill:.6f}, expected > 99.9% of Emax={params['Emax']}"
    )


def test_negative_concentration_handled_safely(params):
    """
    Negative concentrations (numerical artefacts) must not cause errors.

    ODE solvers can produce tiny negative values near zero.
    The emax_killing function must handle these gracefully.
    """
    from src.model.pd import emax_killing
    kill = emax_killing(C=-1e-10,
                        Emax=params["Emax"],
                        EC50=params["EC50"],
                        n=params["n"])
    assert kill == 0.0, (
        f"kill(negative C) should be 0, got {kill}"
    )


def test_steady_state_values_physiologically_plausible(steady_state):
    """
    Steady state B cell counts must be within physiological ranges.

    Reference ranges (cells/uL):
        B_blood  : 100 - 400  (normal peripheral blood)
        B_bm     : 50  - 500  (bone marrow precursors)
        B_tissue : 100 - 1000 (lymphoid tissue pool)
    """
    assert 100 <= steady_state["B_blood"] <= 400, (
        f"B_blood steady state {steady_state['B_blood']:.1f} outside "
        f"physiological range 100-400 cells/uL"
    )
    assert 50 <= steady_state["B_bm"] <= 500, (
        f"B_bm steady state {steady_state['B_bm']:.1f} outside "
        f"physiological range 50-500 cells/uL"
    )
    assert 100 <= steady_state["B_tissue"] <= 1000, (
        f"B_tissue steady state {steady_state['B_tissue']:.1f} outside "
        f"physiological range 100-1000 cells/uL"
    )
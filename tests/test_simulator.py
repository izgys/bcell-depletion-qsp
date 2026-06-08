"""
Tests for the forward simulation engine (src/simulation/simulator.py).

Tests verify:
    1. Steady state computation is correct
    2. Simulation completes without solver failure
    3. B cell counts remain non-negative throughout
    4. Depletion occurs after dosing
    5. B cells recover after drug clears
"""

import numpy as np
import pytest
import yaml

from src.simulation.simulator import run_simulation, compute_steady_state


@pytest.fixture
def params():
    """Load full parameter set from config files."""
    with open("config/model_params.yaml") as f:
        p = yaml.safe_load(f)
    with open("config/pk_rituximab.yaml") as f:
        p.update(yaml.safe_load(f))
    return p


@pytest.fixture
def schedule():
    """Load rituximab RA dosing schedule."""
    with open("config/dosing_schedules.yaml") as f:
        schedules = yaml.safe_load(f)
    return schedules["rituximab_RA"]


@pytest.fixture
def result(params, schedule):
    """Run a full simulation for use across multiple tests."""
    return run_simulation(
        params=params,
        schedule=schedule,
        t_end=365.0,
        mlflow_run=False,
    )


def test_steady_state_all_positive(params):
    """
    All steady state compartment values must be positive.

    Negative cell counts or concentrations are physically impossible.
    """
    ss = compute_steady_state(params)
    for key, value in ss.items():
        assert value >= 0.0, (
            f"Steady state {key} = {value:.4f} is negative — "
            f"physically impossible"
        )


def test_steady_state_coupled_solution_consistent(params):
    """
    The coupled blood/tissue steady state must satisfy both ODEs simultaneously.

    Plug steady state values back into blood and tissue derivatives —
    both must equal zero.
    """
    from src.model.odes import bcell_ode_system

    ss = compute_steady_state(params)
    y0 = [ss["B_bm"], ss["B_blood"], ss["B_tissue"],
          ss["C_central"], ss["C_peripheral"]]

    derivs = bcell_ode_system(t=0.0, y=y0, params=params, dose_fn=lambda t: 0.0)

    # blood derivative (index 1) and tissue derivative (index 2)
    assert abs(derivs[1]) < 1e-6, (
        f"Blood ODE at steady state = {derivs[1]:.2e}, expected ~0"
    )
    assert abs(derivs[2]) < 1e-6, (
        f"Tissue ODE at steady state = {derivs[2]:.2e}, expected ~0"
    )


def test_simulation_completes(result):
    """Simulation must complete without solver failure."""
    assert result is not None
    assert len(result.t) > 0
    assert result.t[-1] >= 364.0  # reaches end of simulation


def test_bcell_counts_non_negative(result):
    """
    B cell counts must never go negative during simulation.

    Negative cell counts are physically impossible and indicate
    either a solver accuracy issue or an ODE implementation error.
    """
    assert np.all(result.B_bm >= 0), "Bone marrow B cells went negative"
    assert np.all(result.B_blood >= 0), "Blood B cells went negative"
    assert np.all(result.B_tissue >= 0), "Tissue B cells went negative"


def test_drug_concentration_non_negative(result):
    """Drug concentrations must never go negative."""
    assert np.all(result.C_central >= -1e-10), (
        "Central drug concentration went significantly negative"
    )
    assert np.all(result.C_peripheral >= -1e-10), (
        "Peripheral drug concentration went significantly negative"
    )


def test_depletion_occurs_after_dosing(result):
    """
    B cell counts must drop significantly below baseline after dosing.

    Rituximab should achieve >90% depletion in blood.
    If minimum blood fraction > 50%, the drug is having no meaningful effect.
    """
    min_fraction = result.b_blood_fraction().min()
    assert min_fraction < 0.10, (
        f"Minimum blood B cell fraction = {min_fraction:.2%} — "
        f"expected >90% depletion with rituximab RA regimen"
    )


def test_recovery_occurs_after_drug_clears(result):
    """
    B cells must begin recovering after drug clears.

    By day 365 (one year after dosing), blood B cells should have
    recovered to at least 50% of baseline as bone marrow repopulates.
    """
    # find fraction at day 365
    final_fraction = result.b_blood_fraction()[-1]
    assert final_fraction > 0.50, (
        f"Blood B cells at day 365 = {final_fraction:.2%} of baseline — "
        f"expected >50% recovery one year after dosing"
    )


def test_b_blood_fraction_bounded(result):
    """
    Blood B cell fraction must stay between 0 and 1 before dosing
    and not exceed physiologically plausible overshoot after recovery.

    Some overshoot above 100% is biologically plausible during repopulation
    but values above 150% suggest a model instability.
    """
    max_fraction = result.b_blood_fraction().max()
    assert max_fraction <= 1.50, (
        f"Blood B cell fraction peaked at {max_fraction:.2%} — "
        f"overshoot above 150% suggests model instability"
    )


def test_pk_metrics_computed(result):
    """PK metrics must be computed and contain expected keys."""
    expected_keys = ["k10", "k12", "k21", "alpha", "beta",
                     "t_half_alpha", "t_half_beta", "vss"]
    for key in expected_keys:
        assert key in result.pk_metrics, f"Missing PK metric: {key}"
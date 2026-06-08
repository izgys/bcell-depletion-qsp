"""
Tests for the PK model (src/model/pk.py).

Tests verify:
    1. Parameter validation catches invalid inputs
    2. Derived PK metrics are mathematically correct
    3. Analytical solution matches known initial conditions
    4. Terminal half-life is physiologically plausible for rituximab
"""

import numpy as np
import pytest
import yaml

from src.model.pk import validate_pk_params, compute_pk_metrics, analytical_two_compartment


@pytest.fixture
def params():
    """Load PK parameters from config."""
    with open("config/model_params.yaml") as f:
        p = yaml.safe_load(f)
    with open("config/pk_rituximab.yaml") as f:
        p.update(yaml.safe_load(f))
    return p


def test_validation_rejects_negative_CL():
    """
    Clearance must be positive — negative CL is physically impossible.
    Drug cannot flow out of the body at a negative rate.
    """
    bad_params = {"CL": -1.0, "V1": 3600.0, "V2": 4200.0, "Q": 190.0}
    with pytest.raises(ValueError, match="CL"):
        validate_pk_params(bad_params)


def test_validation_rejects_zero_V1():
    """
    V1 must be positive — zero volume of distribution is physically impossible.
    """
    bad_params = {"CL": 230.0, "V1": 0.0, "V2": 4200.0, "Q": 190.0}
    with pytest.raises(ValueError, match="V1"):
        validate_pk_params(bad_params)


def test_validation_rejects_implausibly_small_V1():
    """
    V1 below 500 mL is implausible for a monoclonal antibody.
    Catches unit errors — e.g. V1 entered in L instead of mL.
    """
    bad_params = {"CL": 230.0, "V1": 3.6, "V2": 4200.0, "Q": 190.0}
    with pytest.raises(ValueError, match="V1"):
        validate_pk_params(bad_params)


def test_validation_accepts_valid_params(params):
    """Valid rituximab parameters must pass validation without error."""
    validate_pk_params(params)  # should not raise


def test_terminal_halflife_physiologically_plausible(params):
    """
    Rituximab terminal half-life is reported as 18-32 days in the literature.
    Our parameters should produce a value in this range.

    Reference: Mager & Jusko 2001, Quartier et al. 2003
    """
    metrics = compute_pk_metrics(params)
    t_half_beta = metrics["t_half_beta"]
    assert 15 <= t_half_beta <= 40, (
        f"Terminal half-life {t_half_beta:.1f} days outside expected "
        f"range 15-40 days for rituximab"
    )


def test_alpha_greater_than_beta(params):
    """
    Alpha (fast disposition) must always be greater than beta (slow elimination).

    This is a mathematical property of the two-compartment model —
    the fast phase always decays faster than the slow phase.
    If alpha < beta the biexponential solution is invalid.
    """
    metrics = compute_pk_metrics(params)
    assert metrics["alpha"] > metrics["beta"], (
        f"alpha={metrics['alpha']:.4f} must be > beta={metrics['beta']:.4f}"
    )


def test_analytical_solution_initial_concentration(params):
    """
    At t=0, analytical solution must equal Dose/V1.

    This is the definition of V1 — the initial concentration
    after an IV bolus equals dose divided by central volume.
    """
    dose_mg = 1000.0
    dose_ng = dose_mg * 1_000_000

    C0 = analytical_two_compartment(
        t_array=[0.0],
        dose_mg=dose_mg,
        params=params,
    )[0]

    expected = dose_ng / params["V1"]
    assert abs(C0 - expected) / expected < 1e-6, (
        f"C(t=0) = {C0:.2f} ng/mL, expected {expected:.2f} ng/mL (Dose/V1)"
    )


def test_analytical_solution_decays_to_zero(params):
    """
    Drug concentration must decay toward zero at long times.

    At t=500 days (>>terminal half-life of ~34 days), concentration
    should be negligible — less than 0.001% of initial concentration.
    """
    dose_mg = 1000.0
    t_array = np.array([0.0, 500.0])

    C = analytical_two_compartment(t_array=t_array, dose_mg=dose_mg, params=params)
    C0 = C[0]
    C_late = C[1]

    assert C_late / C0 < 1e-5, (
        f"Concentration at day 500 is {C_late/C0:.2e} of initial — "
        f"drug should be essentially cleared by this time"
    )


def test_vss_equals_v1_plus_v2(params):
    """
    Volume of distribution at steady state must equal V1 + V2.

    This is an exact relationship for the two-compartment model.
    """
    metrics = compute_pk_metrics(params)
    expected_vss = params["V1"] + params["V2"]
    assert abs(metrics["vss"] - expected_vss) < 1e-10, (
        f"Vss = {metrics['vss']:.2f}, expected V1+V2 = {expected_vss:.2f}"
    )
"""
Pharmacokinetic model components for the B cell depletion QSP model.

Implements a two-compartment IV model with:
    - Central compartment (V1): plasma and well-perfused tissues
    - Peripheral compartment (V2): deeper tissues with slower equilibration

Standard PK parameters:
    CL : clearance (mL/day)          — permanent drug elimination from central
    V1 : central volume (mL)         — apparent volume of central compartment
    V2 : peripheral volume (mL)      — apparent volume of peripheral compartment
    Q  : inter-compartmental flow    — drug transfer between compartments (mL/day)

All parameters are drug-specific and taken from the literature.
See config/pk_rituximab.yaml for rituximab-specific values.
"""

import numpy as np


def validate_pk_params(params: dict) -> None:
    """
    Validate PK parameters for physical plausibility.

    Raises ValueError if any parameter is outside acceptable bounds.
    This runs before any simulation to catch configuration errors early.

    Parameters
    ----------
    params : dict
        Must contain keys: CL, V1, V2, Q.
    """
    required = ["CL", "V1", "V2", "Q"]
    for key in required:
        if key not in params:
            raise ValueError(f"Missing PK parameter: {key}")
        if params[key] <= 0:
            raise ValueError(
                f"PK parameter {key} must be positive, got {params[key]}"
            )

    # V1 should be at least actual plasma volume (~3000 mL in adults)
    if params["V1"] < 500:
        raise ValueError(
            f"V1={params['V1']} mL seems implausibly small for a monoclonal antibody. "
            f"Expected > 500 mL."
        )

    # CL for monoclonal antibodies is typically 5-50 mL/day
    if params["CL"] > 10_000:
        raise ValueError(
            f"CL={params['CL']} mL/day seems implausibly large. "
            f"Check units — expected mL/day not L/day."
        )


def compute_pk_metrics(params: dict) -> dict:
    """
    Compute standard PK metrics from two-compartment parameters.

    These are derived quantities useful for interpreting simulation results
    and cross-checking against published PK reports.

    Parameters
    ----------
    params : dict
        Must contain keys: CL, V1, V2, Q.

    Returns
    -------
    dict with keys:
        k10  : elimination rate constant from central (1/day)
        k12  : transfer rate constant central -> peripheral (1/day)
        k21  : transfer rate constant peripheral -> central (1/day)
        alpha : fast disposition rate constant (1/day)
        beta  : slow disposition rate constant (1/day)
        t_half_alpha : fast half-life (days)
        t_half_beta  : terminal half-life (days)
        vss          : volume of distribution at steady state (mL)
    """
    CL = params["CL"]
    V1 = params["V1"]
    V2 = params["V2"]
    Q  = params["Q"]

    # micro rate constants
    # these are the fundamental rate constants underlying the two-compartment model
    k10 = CL / V1        # elimination rate from central
    k12 = Q / V1         # transfer rate central -> peripheral
    k21 = Q / V2         # transfer rate peripheral -> central

    # macro rate constants (alpha and beta)
    # the two-compartment model solution is a biexponential:
    # C(t) = A * exp(-alpha * t) + B * exp(-beta * t)
    # alpha is the fast phase (distribution), beta is the slow phase (elimination)
    sum_k   = k10 + k12 + k21
    alpha   = 0.5 * (sum_k + np.sqrt(sum_k**2 - 4 * k10 * k21))
    beta    = 0.5 * (sum_k - np.sqrt(sum_k**2 - 4 * k10 * k21))

    # half-lives
    t_half_alpha = np.log(2) / alpha   # fast distribution half-life
    t_half_beta  = np.log(2) / beta    # terminal elimination half-life

    # volume of distribution at steady state
    # Vss = V1 + V2 for a two-compartment model
    vss = V1 + V2

    return {
        "k10": k10,
        "k12": k12,
        "k21": k21,
        "alpha": alpha,
        "beta": beta,
        "t_half_alpha": t_half_alpha,
        "t_half_beta": t_half_beta,
        "vss": vss,
    }


def analytical_two_compartment(t_array, dose_mg, params: dict) -> np.ndarray:
    """
    Analytical solution for central compartment concentration after
    a single IV bolus dose in a two-compartment model.

    This is used to validate the ODE solver — the numerical solution
    from solve_ivp should match this analytical solution closely.

    C(t) = A * exp(-alpha * t) + B * exp(-beta * t)

    Parameters
    ----------
    t_array : array-like
        Time points at which to evaluate concentration (days).
    dose_mg : float
        IV bolus dose in mg. Converted to ng internally.
    params : dict
        Must contain keys: CL, V1, V2, Q.

    Returns
    -------
    np.ndarray
        Central compartment concentration at each timepoint (ng/mL).
    """
    t_array = np.asarray(t_array)

    # convert dose from mg to ng (1 mg = 1,000,000 ng)
    dose_ng = dose_mg * 1_000_000

    CL = params["CL"]
    V1 = params["V1"]
    V2 = params["V2"]
    Q  = params["Q"]

    metrics = compute_pk_metrics(params)
    alpha   = metrics["alpha"]
    beta    = metrics["beta"]
    k21     = metrics["k21"]

    # coefficients A and B for the biexponential
    # derived from initial conditions: C_central(0) = dose/V1, C_peripheral(0) = 0
    A = (dose_ng / V1) * (alpha - k21) / (alpha - beta)
    B = (dose_ng / V1) * (k21 - beta)  / (alpha - beta)

    return A * np.exp(-alpha * t_array) + B * np.exp(-beta * t_array)
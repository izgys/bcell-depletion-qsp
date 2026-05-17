"""
Pharmacodynamic models for the B cell depletion QSP model.

Implements the concentration-effect relationship — how drug concentration
translates into B cell killing rate.

The primary model is the sigmoidal Emax (Hill equation):

    kill(C) = Emax * C^n / (EC50^n + C^n)

Where:
    Emax : maximum killing rate (1/day)         — ceiling effect
    EC50 : concentration for 50% max effect     — potency (ng/mL)
    n    : Hill coefficient (dimensionless)      — curve steepness

Important limitations (see README):
    - Emax is the asymptotic ceiling, not necessarily full receptor saturation
    - EC50 always equals Emax/2 mathematically, but does not always correspond
      to 50% receptor occupancy in systems with spare receptors
    - The sigmoidal shape assumes a single binding site population —
      biphasic responses require a sum of two Emax functions
    - Parameters are empirical curve descriptors, not mechanistic quantities,
      unless supported by independent receptor binding data
"""

import numpy as np


def validate_pd_params(params: dict) -> None:
    """
    Validate PD parameters for physical plausibility.

    Raises ValueError if any parameter is outside acceptable bounds.

    Parameters
    ----------
    params : dict
        Must contain keys: Emax, EC50, n.
    """
    required = ["Emax", "EC50", "n"]
    for key in required:
        if key not in params:
            raise ValueError(f"Missing PD parameter: {key}")
        if params[key] <= 0:
            raise ValueError(
                f"PD parameter {key} must be positive, got {params[key]}"
            )

    # Emax is a rate (1/day) — values above 10/day would mean
    # complete B cell pool turnover in under 2.4 hours, implausible
    if params["Emax"] > 10:
        raise ValueError(
            f"Emax={params['Emax']} /day seems implausibly large. "
            f"Expected < 10 /day for anti-CD20 therapy."
        )

    # Hill coefficient n is typically 0.5-5 for biological systems
    # values above 10 produce near-switch behaviour with no biological basis
    if params["n"] > 10:
        raise ValueError(
            f"Hill coefficient n={params['n']} seems implausibly large. "
            f"Expected 0.5-5 for anti-CD20 PD."
        )


def emax_killing(C: float | np.ndarray,
                 Emax: float,
                 EC50: float,
                 n: float) -> float | np.ndarray:
    """
    Sigmoidal Emax killing function (Hill equation).

    Computes the drug-induced B cell killing rate at drug concentration C.

    Mathematical properties:
        - kill(0)    = 0        (no drug, no killing)
        - kill(EC50) = Emax/2   (by definition, always)
        - kill(inf)  = Emax     (asymptotic ceiling)

    Parameters
    ----------
    C : float or np.ndarray
        Drug concentration in central compartment (ng/mL).
        Accepts arrays for vectorised evaluation (e.g. plotting).
    Emax : float
        Maximum killing rate (1/day).
    EC50 : float
        Drug concentration producing 50% of maximum killing (ng/mL).
    n : float
        Hill coefficient — controls steepness of the curve.
        n=1: gradual sigmoid. n>1: steeper transition. n>>1: switch-like.

    Returns
    -------
    float or np.ndarray
        Killing rate at concentration C (1/day).
    """
    C = np.asarray(C, dtype=float)

    # guard against negative concentrations (numerical artefacts near t=0)
    C = np.maximum(C, 0.0)

    return (Emax * C**n) / (EC50**n + C**n)


def linear_killing(C: float | np.ndarray,
                   slope: float) -> float | np.ndarray:
    """
    Linear PD model — killing rate proportional to concentration.

    kill(C) = slope * C

    Simpler alternative to Emax for model comparison.
    Does not saturate — killing rate increases indefinitely with concentration.
    Only appropriate over a narrow concentration range where saturation
    is not expected.

    Parameters
    ----------
    C : float or np.ndarray
        Drug concentration (ng/mL).
    slope : float
        Killing rate per unit concentration (1/day per ng/mL).

    Returns
    -------
    float or np.ndarray
        Killing rate (1/day).
    """
    C = np.asarray(C, dtype=float)
    C = np.maximum(C, 0.0)
    return slope * C


def compute_pd_metrics(Emax: float,
                       EC50: float,
                       n: float) -> dict:
    """
    Compute derived PD metrics for interpretation and reporting.

    Parameters
    ----------
    Emax : float
        Maximum killing rate (1/day).
    EC50 : float
        Concentration for 50% max killing (ng/mL).
    n : float
        Hill coefficient.

    Returns
    -------
    dict with keys:
        EC10  : concentration for 10% max effect (ng/mL)
        EC90  : concentration for 90% max effect (ng/mL)
        EC10_to_EC90_ratio : dynamic range of the curve
            narrow ratio (e.g. 4) means steep transition
            wide ratio (e.g. 100) means gradual transition
    """
    # inverse of Emax equation: EC_x = EC50 * (x / (1-x))^(1/n)
    # where x is the fraction of Emax (e.g. 0.1 for EC10)
    EC10 = EC50 * (0.1 / 0.9) ** (1 / n)
    EC90 = EC50 * (0.9 / 0.1) ** (1 / n)

    return {
        "EC10": EC10,
        "EC90": EC90,
        "EC10_to_EC90_ratio": EC90 / EC10,
    }
"""
ODE system for the B cell depletion QSP model.

State vector (y):
    y[0] = B_bm         — bone marrow B cell precursor pool (cells/uL)
    y[1] = B_blood      — peripheral blood B cells (cells/uL)
    y[2] = B_tissue     — lymphoid tissue B cells (cells/uL)
    y[3] = C_central    — drug concentration, central compartment (ng/mL)
    y[4] = C_peripheral — drug concentration, peripheral compartment (ng/mL)

All rates are in units of 1/day.
All concentrations are in units of ng/mL.
All cell counts are in units of cells/uL.
"""

import numpy as np


def bcell_ode_system(t, y, params, dose_fn):
    """
    Right-hand side of the B cell depletion ODE system.

    Parameters
    ----------
    t : float
        Current time (days). Passed automatically by scipy.
    y : array-like of shape (5,)
        Current state vector [B_bm, B_blood, B_tissue, C_central, C_peripheral].
    params : dict
        Model parameters. See model_params.yaml for full specification.
    dose_fn : callable
        Function dose_fn(t) -> float returning the drug infusion rate at time t
        in units of ng/mL/day into the central compartment.

    Returns
    -------
    list of float
        Derivatives [dB_bm_dt, dB_blood_dt, dB_tissue_dt,
                     dC_central_dt, dC_peripheral_dt].
    """

    # --- unpack state vector ---
    B_bm, B_blood, B_tissue, C_central, C_peripheral = y

    # --- unpack parameters ---
    # B cell parameters
    k_prod       = params["k_prod"]        # bone marrow production rate (cells/uL/day)
    k_export     = params["k_export"]      # BM -> blood export rate (1/day)
    d_bm         = params["d_bm"]          # BM natural death rate (1/day)
    k_tissue     = params["k_tissue"]      # blood -> tissue trafficking rate (1/day)
    k_recirculate= params["k_recirculate"] # tissue -> blood recirculation rate (1/day)
    d_blood      = params["d_blood"]       # blood natural death rate (1/day)
    d_tissue     = params["d_tissue"]      # tissue natural death rate (1/day)
    f_tissue     = params["f_tissue"]      # tissue drug penetration factor (dimensionless, 0-1)

    # PK parameters
    CL = params["CL"]  # clearance (mL/day)
    V1 = params["V1"]  # central volume of distribution (mL)
    V2 = params["V2"]  # peripheral volume of distribution (mL)
    Q  = params["Q"]   # inter-compartmental clearance (mL/day)

    # PD parameters
    Emax = params["Emax"]  # maximum killing rate (1/day)
    EC50 = params["EC50"]  # concentration for 50% max killing (ng/mL)
    n    = params["n"]     # Hill coefficient (dimensionless)

    # --- PD: compute killing rate at current drug concentration ---
    # sigmoidal Emax function
    # kill(C) -> 0 as C -> 0 (no drug, no killing)
    # kill(C) -> Emax as C -> inf (full saturation)
    kill = (Emax * C_central**n) / (EC50**n + C_central**n)

    # --- get current dose rate from dosing function ---
    dose_rate = dose_fn(t)  # ng/mL/day into central compartment

    # --- ODEs ---

    # bone marrow: production - export - natural death
    dB_bm_dt = (k_prod
                - k_export * B_bm
                - d_bm * B_bm)

    # peripheral blood: import from BM + recirculation from tissue
    #                   - trafficking to tissue - natural death - drug killing
    dB_blood_dt = (k_export * B_bm
                   + k_recirculate * B_tissue
                   - k_tissue * B_blood
                   - d_blood * B_blood
                   - kill * B_blood)

    # lymphoid tissue: trafficking from blood
    #                  - recirculation to blood - natural death - drug killing (scaled)
    dB_tissue_dt = (k_tissue * B_blood
                    - k_recirculate * B_tissue
                    - d_tissue * B_tissue
                    - kill * f_tissue * B_tissue)

    # central PK compartment: dose input - elimination - distribution to peripheral
    #                          + return from peripheral
    dC_central_dt = (dose_rate
                     - (CL / V1) * C_central
                     - (Q / V1) * C_central
                     + (Q / V2) * C_peripheral)

    # peripheral PK compartment: distribution from central - return to central
    dC_peripheral_dt = ((Q / V1) * C_central
                        - (Q / V2) * C_peripheral)

    return [dB_bm_dt, dB_blood_dt, dB_tissue_dt,
            dC_central_dt, dC_peripheral_dt]
"""
Forward simulation engine for the B cell depletion QSP model.

Orchestrates the full simulation pipeline:
    1. Validate parameters
    2. Compute steady-state initial conditions analytically
    3. Build dose function from dosing schedule
    4. Integrate ODE system with scipy LSODA solver
    5. Package results into a clean SimulationResult object
    6. Log parameters and outputs to MLflow

Usage:
    from src.simulation.simulator import run_simulation
    import yaml

    with open("config/model_params.yaml") as f:
        params = yaml.safe_load(f)
    with open("config/pk_rituximab.yaml") as f:
        params.update(yaml.safe_load(f))
    with open("config/dosing_schedules.yaml") as f:
        schedule = yaml.safe_load(f)

    result = run_simulation(params, schedule)
"""

import numpy as np
import mlflow
from dataclasses import dataclass
from scipy.integrate import solve_ivp

from src.model.odes import bcell_ode_system
from src.model.pk import validate_pk_params, compute_pk_metrics
from src.model.pd import validate_pd_params
from src.model.dosing import events_from_config, get_discontinuities


@dataclass
class SimulationResult:
    """
    Output of a single forward simulation run.

    Attributes
    ----------
    t : np.ndarray
        Timepoints (days).
    B_bm : np.ndarray
        Bone marrow B cell counts over time (cells/uL).
    B_blood : np.ndarray
        Peripheral blood B cell counts over time (cells/uL).
    B_tissue : np.ndarray
        Lymphoid tissue B cell counts over time (cells/uL).
    C_central : np.ndarray
        Central compartment drug concentration over time (ng/mL).
    C_peripheral : np.ndarray
        Peripheral compartment drug concentration over time (ng/mL).
    steady_state : dict
        Analytically computed pre-dose steady state values.
    pk_metrics : dict
        Derived PK metrics (half-lives, rate constants).
    params : dict
        Full parameter set used for this simulation.
    """
    t: np.ndarray
    B_bm: np.ndarray
    B_blood: np.ndarray
    B_tissue: np.ndarray
    C_central: np.ndarray
    C_peripheral: np.ndarray
    steady_state: dict
    pk_metrics: dict
    params: dict

    def b_blood_fraction(self) -> np.ndarray:
        """
        Blood B cell count as fraction of pre-dose steady state.
        Useful for plotting depletion as % of baseline.
        """
        return self.B_blood / self.steady_state["B_blood"]

    def b_tissue_fraction(self) -> np.ndarray:
        """
        Tissue B cell count as fraction of pre-dose steady state.
        """
        return self.B_tissue / self.steady_state["B_tissue"]


def compute_steady_state(params: dict) -> dict:
    """
    Analytically compute pre-dose steady-state B cell counts.

    At steady state all derivatives are zero and no drug is present.
    Solving the ODE system algebraically gives:

        B_bm*     = k_prod / (k_export + d_bm)
        B_blood*  = (k_export * B_bm* + k_recirculate * B_tissue*) /
                    (k_tissue + d_blood)
        B_tissue* = k_tissue * B_blood* / (k_recirculate + d_tissue)

    Note: B_blood* and B_tissue* are coupled — solved iteratively.
    B_blood* depends on B_tissue* and vice versa. We resolve this
    by substituting B_tissue* into the B_blood* equation and solving
    the resulting linear system analytically.

    Parameters
    ----------
    params : dict
        Model parameters.

    Returns
    -------
    dict with keys: B_bm, B_blood, B_tissue, C_central, C_peripheral
    """
    k_prod        = params["k_prod"]
    k_export      = params["k_export"]
    d_bm          = params["d_bm"]
    k_tissue      = params["k_tissue"]
    k_recirculate = params["k_recirculate"]
    d_blood       = params["d_blood"]
    d_tissue      = params["d_tissue"]

    # bone marrow — independent, solved directly
    B_bm = k_prod / (k_export + d_bm)

    # blood and tissue are coupled:
    # B_blood* = (k_export * B_bm + k_recirculate * B_tissue*) / (k_tissue + d_blood)
    # B_tissue* = k_tissue * B_blood* / (k_recirculate + d_tissue)
    #
    # substitute B_tissue* into B_blood* equation:
    # B_blood* = (k_export * B_bm + k_recirculate * k_tissue * B_blood* /
    #            (k_recirculate + d_tissue)) / (k_tissue + d_blood)
    #
    # rearrange to isolate B_blood*:
    # B_blood* * (k_tissue + d_blood) = k_export * B_bm +
    #            k_recirculate * k_tissue * B_blood* / (k_recirculate + d_tissue)
    #
    # B_blood* * [(k_tissue + d_blood) -
    #            k_recirculate * k_tissue / (k_recirculate + d_tissue)]
    #          = k_export * B_bm
    #
    # solve for B_blood*:
    denom_blood = (
        (k_tissue + d_blood)
        - (k_recirculate * k_tissue) / (k_recirculate + d_tissue)
    )
    B_blood = (k_export * B_bm) / denom_blood

    # tissue — now solved directly from B_blood*
    B_tissue = (k_tissue * B_blood) / (k_recirculate + d_tissue)

    return {
        "B_bm": B_bm,
        "B_blood": B_blood,
        "B_tissue": B_tissue,
        "C_central": 0.0,
        "C_peripheral": 0.0,
    }


def run_simulation(
    params: dict,
    schedule: dict,
    t_end: float = 365.0,
    n_timepoints: int = 1000,
    mlflow_run: bool = True,
) -> SimulationResult:
    """
    Run a full forward simulation of the B cell depletion model.

    Parameters
    ----------
    params : dict
        Combined model parameters (physiological + PK + PD).
        Loaded from config/model_params.yaml and config/pk_*.yaml.
    schedule : dict
        Dosing schedule loaded from config/dosing_schedules.yaml.
    t_end : float
        Simulation end time in days. Default 365 (one year).
    n_timepoints : int
        Number of timepoints in output. Default 1000.
    mlflow_run : bool
        Whether to log this run to MLflow. Default True.

    Returns
    -------
    SimulationResult
        Structured simulation output.
    """
    # --- step 1: validate parameters ---
    validate_pk_params(params)
    validate_pd_params(params)

    # --- step 2: compute steady-state initial conditions ---
    steady_state = compute_steady_state(params)
    y0 = [
        steady_state["B_bm"],
        steady_state["B_blood"],
        steady_state["B_tissue"],
        steady_state["C_central"],
        steady_state["C_peripheral"],
    ]

    # --- step 3: build dose function from schedule ---
    events, dose_fn = events_from_config(schedule, params["V1"])

    # get discontinuity timepoints so solver steps precisely at dose events
    discontinuities = get_discontinuities(events)

    # --- step 4: define time span and evaluation points ---
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, n_timepoints)

    # add discontinuities to t_eval so they appear in output
    t_eval = np.union1d(t_eval, discontinuities)
    t_eval = t_eval[t_eval <= t_end]

    # --- step 5: run ODE solver ---
    # LSODA: automatically switches between stiff and non-stiff methods
    # this handles the fast PK timescale alongside slow B cell timescale
    sol = solve_ivp(
        fun=lambda t, y: bcell_ode_system(t, y, params, dose_fn),
        t_span=t_span,
        y0=y0,
        method="LSODA",
        t_eval=t_eval,
        dense_output=False,
        # discontinuities: force solver to step at dose start/end times
        # prevents solver from stepping over infusion events
        events=None,
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(
            f"ODE solver failed: {sol.message}\n"
            f"Consider checking parameter values or reducing rtol/atol."
        )

    # --- step 6: package results ---
    pk_metrics = compute_pk_metrics(params)

    result = SimulationResult(
        t=sol.t,
        B_bm=sol.y[0],
        B_blood=sol.y[1],
        B_tissue=sol.y[2],
        C_central=sol.y[3],
        C_peripheral=sol.y[4],
        steady_state=steady_state,
        pk_metrics=pk_metrics,
        params=params,
    )

    # --- step 7: log to MLflow ---
    if mlflow_run:
        _log_to_mlflow(result, schedule)

    return result


def _log_to_mlflow(result: SimulationResult, schedule: dict) -> None:
    """
    Log simulation parameters and key outputs to MLflow.

    Logs:
        - All model parameters as MLflow params
        - Steady-state B cell counts as metrics
        - PK metrics (half-lives) as metrics
        - Minimum B cell counts during simulation as metrics
        - Time to nadir (lowest B cell count) as metrics

    Parameters
    ----------
    result : SimulationResult
    schedule : dict
        Dosing schedule for logging.
    """
    with mlflow.start_run(nested=True):
        # log all parameters
        for key, value in result.params.items():
            mlflow.log_param(key, value)

        # log dosing schedule summary
        mlflow.log_param("n_doses", len(schedule["doses"]))
        mlflow.log_param(
            "total_dose_mg",
            sum(d["dose_mg"] for d in schedule["doses"])
        )

        # log steady-state values
        for key, value in result.steady_state.items():
            mlflow.log_metric(f"ss_{key}", value)

        # log PK metrics
        mlflow.log_metric("t_half_beta_days", result.pk_metrics["t_half_beta"])
        mlflow.log_metric("t_half_alpha_days", result.pk_metrics["t_half_alpha"])

        # log depletion metrics
        min_blood_fraction = result.b_blood_fraction().min()
        min_tissue_fraction = result.b_tissue_fraction().min()
        nadir_day_blood = result.t[result.b_blood_fraction().argmin()]
        nadir_day_tissue = result.t[result.b_tissue_fraction().argmin()]

        mlflow.log_metric("min_blood_fraction", min_blood_fraction)
        mlflow.log_metric("min_tissue_fraction", min_tissue_fraction)
        mlflow.log_metric("nadir_day_blood", nadir_day_blood)
        mlflow.log_metric("nadir_day_tissue", nadir_day_tissue)
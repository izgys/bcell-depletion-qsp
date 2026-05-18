"""
Trajectory visualisation for the B cell depletion QSP model.

Produces publication-quality plots of simulation outputs:
    - B cell counts over time (all compartments)
    - B cell depletion as fraction of baseline
    - Drug concentration over time (central + peripheral)
    - Combined PK/PD summary figure

All functions accept a SimulationResult and return a matplotlib Figure.
Figures are saved to the output path if provided, and logged to MLflow
as artifacts if an active run exists.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlflow

from src.simulation.simulator import SimulationResult


# --- plot styling ---
COLORS = {
    "blood":      "#2196F3",   # blue
    "tissue":     "#4CAF50",   # green
    "bm":         "#FF9800",   # orange
    "central":    "#9C27B0",   # purple
    "peripheral": "#E91E63",   # pink
    "dose":       "#F44336",   # red
}


def _add_dose_lines(ax, schedule: dict, label: bool = True) -> None:
    """
    Add vertical dashed lines at each dose start time.

    Parameters
    ----------
    ax : matplotlib Axes
    schedule : dict
        Dosing schedule from config.
    label : bool
        Whether to add 'Dose' label to first line.
    """
    for i, dose in enumerate(schedule["doses"]):
        ax.axvline(
            x=dose["start_day"],
            color=COLORS["dose"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="Dose" if (i == 0 and label) else None,
        )


def plot_bcell_trajectories(
    result: SimulationResult,
    schedule: dict,
    output_path: str = None,
) -> plt.Figure:
    """
    Plot B cell counts over time for all three compartments.

    Parameters
    ----------
    result : SimulationResult
    schedule : dict
        Dosing schedule — used to mark dose timepoints.
    output_path : str, optional
        Path to save figure. If None, figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- top panel: absolute B cell counts ---
    ax = axes[0]
    ax.plot(result.t, result.B_blood,
            color=COLORS["blood"], linewidth=2, label="Blood")
    ax.plot(result.t, result.B_tissue,
            color=COLORS["tissue"], linewidth=2, label="Lymphoid tissue")
    ax.plot(result.t, result.B_bm,
            color=COLORS["bm"], linewidth=2, label="Bone marrow", linestyle="--")

    _add_dose_lines(ax, schedule)
    ax.set_ylabel("B cells (cells/µL)", fontsize=11)
    ax.set_title("B Cell Dynamics Under Anti-CD20 Therapy", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # --- bottom panel: fraction of baseline ---
    ax = axes[1]
    ax.plot(result.t, result.b_blood_fraction() * 100,
            color=COLORS["blood"], linewidth=2, label="Blood")
    ax.plot(result.t, result.b_tissue_fraction() * 100,
            color=COLORS["tissue"], linewidth=2, label="Lymphoid tissue")

    # add nadir annotation
    nadir_idx = result.b_blood_fraction().argmin()
    nadir_t = result.t[nadir_idx]
    nadir_val = result.b_blood_fraction()[nadir_idx] * 100
    ax.annotate(
        f"Nadir: {nadir_val:.1f}%\n(day {nadir_t:.0f})",
        xy=(nadir_t, nadir_val),
        xytext=(nadir_t + 20, nadir_val + 10),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    _add_dose_lines(ax, schedule, label=False)
    ax.set_ylabel("B cells (% of baseline)", fontsize=11)
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 120)
    ax.axhline(y=100, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        # log to MLflow if active run exists
        try:
            mlflow.log_artifact(output_path)
        except Exception:
            pass

    return fig


def plot_pk_trajectory(
    result: SimulationResult,
    schedule: dict,
    output_path: str = None,
) -> plt.Figure:
    """
    Plot drug concentration over time in both compartments.

    Parameters
    ----------
    result : SimulationResult
    schedule : dict
    output_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(result.t, result.C_central,
            color=COLORS["central"], linewidth=2, label="Central (plasma)")
    ax.plot(result.t, result.C_peripheral,
            color=COLORS["peripheral"], linewidth=2,
            label="Peripheral (tissue)", linestyle="--")

    _add_dose_lines(ax, schedule)
    ax.set_ylabel("Drug concentration (ng/mL)", fontsize=11)
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_title("Rituximab PK — Two-Compartment Model", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        try:
            mlflow.log_artifact(output_path)
        except Exception:
            pass

    return fig


def plot_summary(
    result: SimulationResult,
    schedule: dict,
    output_path: str = None,
) -> plt.Figure:
    """
    Combined summary figure — B cell depletion + PK in one plot.

    Three panels:
        1. B cell fraction of baseline (blood + tissue)
        2. Drug concentration (central + peripheral)
        3. Emax killing rate over time

    Parameters
    ----------
    result : SimulationResult
    schedule : dict
    output_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from src.model.pd import emax_killing

    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.35)

    # --- panel 1: B cell depletion ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(result.t, result.b_blood_fraction() * 100,
             color=COLORS["blood"], linewidth=2, label="Blood B cells")
    ax1.plot(result.t, result.b_tissue_fraction() * 100,
             color=COLORS["tissue"], linewidth=2, label="Tissue B cells")
    _add_dose_lines(ax1, schedule)
    ax1.set_ylabel("B cells (% baseline)", fontsize=10)
    ax1.set_title("QSP Model — Rituximab RA Regimen (1000mg × 2, 14 days apart)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 120)
    ax1.axhline(y=100, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # --- panel 2: drug concentration ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(result.t, result.C_central,
             color=COLORS["central"], linewidth=2, label="Central (plasma)")
    ax2.plot(result.t, result.C_peripheral,
             color=COLORS["peripheral"], linewidth=2,
             label="Peripheral", linestyle="--")
    _add_dose_lines(ax2, schedule, label=False)
    ax2.set_ylabel("Concentration (ng/mL)", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    # --- panel 3: killing rate over time ---
    ax3 = fig.add_subplot(gs[2])
    kill_rate = emax_killing(
        result.C_central,
        result.params["Emax"],
        result.params["EC50"],
        result.params["n"],
    )
    ax3.plot(result.t, kill_rate,
             color="#795548", linewidth=2, label="kill(C) — Emax PD")
    ax3.axhline(
        y=result.params["Emax"],
        color="gray", linestyle=":", alpha=0.7,
        label=f"Emax = {result.params['Emax']} /day"
    )
    _add_dose_lines(ax3, schedule, label=False)
    ax3.set_ylabel("Killing rate (1/day)", fontsize=10)
    ax3.set_xlabel("Time (days)", fontsize=10)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        try:
            mlflow.log_artifact(output_path)
        except Exception:
            pass

    return fig
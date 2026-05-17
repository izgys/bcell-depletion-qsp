"""
Dosing event handler for the B cell depletion QSP model.

Implements IV infusion dosing as a time-varying forcing function
compatible with scipy's solve_ivp.

Dosing is represented as a list of infusion events, each defined by:
    - start_day   : day on which infusion begins
    - dose_mg     : total dose in mg
    - duration_h  : infusion duration in hours

The dose_fn(t) function returns the instantaneous infusion rate
in ng/mL/day into the central compartment at time t (days).

Unit conversion:
    dose_mg [mg] -> dose_ng [ng] = dose_mg * 1e6
    duration_h [hours] -> duration_day [days] = duration_h / 24
    rate [ng/day] = dose_ng / duration_day
    concentration rate [ng/mL/day] = rate / V1

Standard rituximab regimens (see config/dosing_schedules.yaml):
    RA  : 1000mg on day 0 and day 14, repeated every 6 months
    MS  : 600mg on day 0 and day 14, repeated every 6 months
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class InfusionEvent:
    """
    A single IV infusion event.

    Attributes
    ----------
    start_day : float
        Day on which infusion begins (day 0 = first dose).
    dose_mg : float
        Total dose administered during this infusion (mg).
    duration_h : float
        Duration of infusion (hours). Typical: 4-6h for rituximab.
    """
    start_day: float
    dose_mg: float
    duration_h: float

    @property
    def end_day(self) -> float:
        """Day on which infusion ends."""
        return self.start_day + self.duration_h / 24

    @property
    def dose_ng(self) -> float:
        """Total dose in nanograms."""
        return self.dose_mg * 1_000_000

    @property
    def duration_days(self) -> float:
        """Infusion duration in days."""
        return self.duration_h / 24


def build_dose_fn(events: list[InfusionEvent], V1: float):
    """
    Build a dose function compatible with scipy's solve_ivp.

    Returns a callable dose_fn(t) that gives the drug infusion rate
    into the central compartment at time t in ng/mL/day.

    Parameters
    ----------
    events : list of InfusionEvent
        All infusion events in the simulation.
    V1 : float
        Central compartment volume (mL). Used to convert
        ng/day -> ng/mL/day.

    Returns
    -------
    callable
        dose_fn(t: float) -> float
        Returns infusion rate in ng/mL/day at time t.
    """
    def dose_fn(t: float) -> float:
        """
        Infusion rate at time t (ng/mL/day).

        Iterates over all events and returns the combined rate
        from any currently active infusions.
        """
        rate = 0.0
        for event in events:
            if event.start_day <= t < event.end_day:
                # rate in ng/day for this event
                rate_ng_per_day = event.dose_ng / event.duration_days
                # convert to ng/mL/day by dividing by V1
                rate += rate_ng_per_day / V1
        return rate

    return dose_fn


def events_from_config(schedule: dict, V1: float):
    """
    Build infusion events and dose function from a config dictionary.

    Config format (from dosing_schedules.yaml):
        doses:
          - start_day: 0
            dose_mg: 1000
            duration_h: 4.5
          - start_day: 14
            dose_mg: 1000
            duration_h: 4.5

    Parameters
    ----------
    schedule : dict
        Dosing schedule loaded from YAML config.
    V1 : float
        Central compartment volume (mL).

    Returns
    -------
    tuple of (list[InfusionEvent], callable)
        events : list of InfusionEvent objects
        dose_fn : callable dose_fn(t) -> float
    """
    events = [
        InfusionEvent(
            start_day=dose["start_day"],
            dose_mg=dose["dose_mg"],
            duration_h=dose["duration_h"],
        )
        for dose in schedule["doses"]
    ]

    dose_fn = build_dose_fn(events, V1)
    return events, dose_fn


def get_discontinuities(events: list[InfusionEvent]) -> list[float]:
    """
    Return all timepoints where the dose function is discontinuous.

    These are the start and end times of every infusion event.
    Passing these to solve_ivp as 't_eval' breakpoints helps the
    solver handle the discontinuities accurately — without them,
    the solver may step over a dose start or end and introduce error.

    Parameters
    ----------
    events : list of InfusionEvent

    Returns
    -------
    list of float
        Sorted list of discontinuity timepoints (days).
    """
    times = []
    for event in events:
        times.append(event.start_day)
        times.append(event.end_day)
    return sorted(times)
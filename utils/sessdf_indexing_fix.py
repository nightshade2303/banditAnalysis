"""Helpers for safe lag-window indexing on sessdf.

The key idea is to avoid arithmetic on the dataframe index and instead use
explicit row positions after sorting by the true trial order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_sessdf_positions(sessdf: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of sessdf with stable row-position columns.

    The returned dataframe is sorted by animal, session, and trial so that any
    downstream lag logic can use positional indexing safely.
    """

    ordered = sessdf.sort_values(["animal", "session", "trial"]).reset_index(drop=True).copy()
    ordered["animal_row"] = ordered.groupby("animal").cumcount()
    ordered["session_row"] = ordered.groupby(["animal", "session"]).cumcount()
    return ordered


def build_lagged_port_arrays(
    animal_df: pd.DataFrame,
    least_time_bin: int,
    most_time_bin: int,
    prev_trial_lag: int,
    next_trial_lag: int,
    metric_col: str = "port",
    time_col: str = "iti",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build lagged port windows using positional slicing within each session.

    Parameters
    ----------
    animal_df:
        A dataframe for one animal, already prepared with
        ``prepare_sessdf_positions``.
    least_time_bin:
        Filter on ``iti > time_bin`` before building the lag windows.
    most_time_bin:
        Filter on ``iti <= time_bin`` before building the lag windows.
    prev_trial_lag, next_trial_lag:
        Number of trials to include before and after the current trial.
    metric_col:
        Column to extract, typically ``port``.
    """

    iti_trial = animal_df.loc[(animal_df[time_col] <= most_time_bin) & (animal_df[time_col]>least_time_bin)].copy()
    prev_ports = np.full((len(iti_trial), prev_trial_lag), np.nan)
    next_ports = np.full((len(iti_trial), next_trial_lag), np.nan)

    for out_idx, (_, trial_row) in enumerate(iti_trial.iterrows()):
        session_trials = animal_df.loc[animal_df["session"] == trial_row["session"]].reset_index(drop=True)
        pos = int(trial_row["session_row"])

        prev_window = session_trials.iloc[max(0, pos - prev_trial_lag):pos][metric_col].to_numpy()
        next_window = session_trials.iloc[pos:pos + next_trial_lag][metric_col].to_numpy()

        if prev_window.size:
            prev_ports[out_idx, -prev_window.size:] = prev_window
        if next_window.size:
            next_ports[out_idx, :next_window.size] = next_window

    return iti_trial, prev_ports, next_ports
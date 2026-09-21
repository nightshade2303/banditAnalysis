"""Faster helpers for sessdf lag-window construction.

This module mirrors the positional lag logic in ``sessdf_indexing_fix`` but
avoids rebuilding the same per-session dataframe slices for every trial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_sessdf_positions(sessdf: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of sessdf with stable row-position columns."""

    ordered = sessdf.sort_values(["animal", "session", "trial"]).reset_index(drop=True).copy()
    ordered["animal_row"] = ordered.groupby("animal").cumcount()
    ordered["session_row"] = ordered.groupby(["animal", "session"]).cumcount()
    return ordered


def build_session_cache(
    animal_df: pd.DataFrame,
    metric_col: str = "port",
) -> dict[object, np.ndarray]:
    """Cache per-session metric arrays for fast lag slicing."""

    return {
        session: group[metric_col].to_numpy()
        for session, group in animal_df.groupby("session", sort=False)
    }


def build_lagged_port_arrays_fast(
    animal_df: pd.DataFrame,
    least_time_bin: int,
    most_time_bin: int,
    prev_trial_lag: int,
    next_trial_lag: int,
    metric_col: str = "port",
    time_col: str = "iti",
    session_cache: dict[object, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build lagged port windows without repeated dataframe filtering.

    The input dataframe should already have session-relative positions from
    ``prepare_sessdf_positions``.
    """

    iti_trial = animal_df.loc[
        (animal_df[time_col] <= most_time_bin) & (animal_df[time_col] > least_time_bin)
    ].copy()

    prev_ports = np.full((len(iti_trial), prev_trial_lag), np.nan)
    next_ports = np.full((len(iti_trial), next_trial_lag), np.nan)

    if session_cache is None:
        session_cache = build_session_cache(animal_df, metric_col=metric_col)

    for out_idx, trial_row in enumerate(iti_trial.itertuples(index=False)):
        session_values = session_cache[getattr(trial_row, "session")]
        pos = int(getattr(trial_row, "session_row"))

        prev_window = session_values[max(0, pos - prev_trial_lag) : pos]
        next_window = session_values[pos : pos + next_trial_lag]

        if prev_window.size:
            prev_ports[out_idx, -prev_window.size :] = prev_window
        if next_window.size:
            next_ports[out_idx, : next_window.size] = next_window

    return iti_trial, prev_ports, next_ports
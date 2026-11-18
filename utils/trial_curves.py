"""Trial-by-trial curves: one row per rat, one column per trial (1, 2, ...)."""
import warnings

import numpy as np
import pandas as pd

from utils.config import N_TRIALS, SMOOTH, SMOOTH_FULL_WINDOWS, SMOOTH_WINDOW, SMOOTH_WITHIN_SEGMENTS
from utils.dfLoading import subset_trials


def block_trial_matrix(df, col, n_trials=N_TRIALS):
    """Rats x trials: mean of `col` at each of the first n_trials of a block, over the rat's blocks with at
    least n_trials trials. NaN values, e.g. switch distance on stay trials, are left out."""
    blocks = subset_trials(df, n_trials, head_trials=n_trials)
    trial_in_block = blocks.groupby(['animal', 'block_num']).cumcount()
    values = pd.to_numeric(blocks[col], errors='coerce')
    per_rat = values.groupby([blocks.animal, trial_in_block]).mean().unstack()
    return _trial_columns(per_rat)


def segment_matrix(df, col, segments, session_col='session'):
    """Rats x trials for consecutive segments of blocks in the same session (notebook cells 44, 103, 135).

    segments: [(block of the session, 'first' or 'last', number of trials), ...] in x-axis order. A session
    counts only if each block has enough trials for all its segments without overlap, e.g. block 1 >= 125
    and block 2 >= 100 trials for BLOCK_1_TO_2. Each rat's curve is the mean over its sessions."""
    need = {}
    for block, _, n in segments:
        need[block] = need.get(block, 0) + n
    curves = {}
    for animal, rat in df.groupby('animal'):
        rows = []
        for _, sess in rat.groupby(session_col):
            blocks = {b: sess.loc[sess.block_group == b, col].to_numpy(dtype=float) for b in need}
            if all(len(blocks[b]) >= need[b] for b in need):
                rows.append(np.concatenate([blocks[b][:n] if which == 'first' else blocks[b][-n:]
                                            for b, which, n in segments]))
        if rows:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                curves[animal] = np.nanmean(rows, axis=0)
    return _trial_columns(pd.DataFrame.from_dict(curves, orient='index'))


def segment_edges(segments):
    """Column where each segment starts, plus the total number of columns."""
    return np.cumsum([0] + [n for _, _, n in segments])


def smooth_curves(curves, segments=None, window=SMOOTH_WINDOW, smooth=SMOOTH, full_windows=SMOOTH_FULL_WINDOWS):
    """Centred moving average along trials that ignores NaN. With full_windows, the trials whose window runs
    past an edge (the first and last window // 2 trials) are NaN; otherwise they average over the trials
    available (notebook cell 103). With segments and SMOOTH_WITHIN_SEGMENTS, segments are smoothed apart, so
    each segment has its own edges."""
    if not smooth:
        return curves
    values = curves.to_numpy(dtype=float)
    edges = segment_edges(segments) if segments is not None and SMOOTH_WITHIN_SEGMENTS else [0, values.shape[1]]
    out = np.full_like(values, np.nan)
    kernel = np.ones(window)
    for start, end in zip(edges[:-1], edges[1:]):
        seg = values[:, start:end]
        valid = ~np.isnan(seg)
        total = np.array([np.convolve(row, kernel, 'same') for row in np.where(valid, seg, 0)])
        count = np.array([np.convolve(row, kernel, 'same') for row in valid.astype(float)])
        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, start:end] = total / count
        if full_windows:
            half = window // 2
            out[:, start:start + half] = np.nan
            out[:, max(end - half, start):end] = np.nan
    return pd.DataFrame(out, index=curves.index, columns=curves.columns)


def _trial_columns(curves):
    """Name the axes of a rats x trials table: index 'animal', columns 'trial' numbered from 1."""
    curves.columns = np.arange(1, curves.shape[1] + 1)
    curves.index.name, curves.columns.name = 'animal', 'trial'
    return curves

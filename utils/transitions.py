"""Choice transition matrices: P(port at trial t+1 | port at trial t)."""
import warnings

import numpy as np
import pandas as pd

from utils.config import PORTS, TM_AGG


def rat_transition_matrices(df, reward=None):
    """{rat: 4 x 4 matrix}, rows = port at t, columns = port at t+1, each row sums to 1 (notebook cell 19).
    A port the rat never chose gives a NaN row. reward=1 or 0 uses only trials with that outcome (cell 21)."""
    if reward is not None:
        df = df[df.reward == reward]
    mats = {}
    for animal, g in df.dropna(subset=['choice_t1']).groupby('animal'):
        counts = pd.crosstab(g.port, g.choice_t1.astype(int)).reindex(index=PORTS, columns=PORTS, fill_value=0)
        mats[animal] = counts.div(counts.sum(axis=1), axis=0).to_numpy()
    return mats


def cohort_transition_matrix(df, reward=None, agg=TM_AGG):
    """Transition matrix combined over rats ('mean' or 'median', ignoring NaN rows), and the number of rats."""
    mats = rat_transition_matrices(df, reward)
    return _combine(list(mats.values()), agg), len(mats)


def cohort_reward_difference(df, agg=TM_AGG):
    """Rewarded minus unrewarded transition matrix of each rat, combined over rats (notebook cell 21)."""
    rewarded, unrewarded = rat_transition_matrices(df, 1), rat_transition_matrices(df, 0)
    diffs = [rewarded[a] - unrewarded[a] for a in rewarded if a in unrewarded]
    return _combine(diffs, agg), len(diffs)


def _combine(mats, agg):
    combine = {'mean': np.nanmean, 'median': np.nanmedian}[agg]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # cells where every rat has NaN
        return combine(np.stack(mats), axis=0)

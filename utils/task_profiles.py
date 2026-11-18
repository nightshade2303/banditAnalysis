"""The task's reward profiles: reward % at ports 1-4 in a block.

Every block uses one of two sets of values, VALUE_SETS, in some port order. The 4 structured (correlated)
profiles are STR_PROFILES; the 32 shuffled ones, UNSTR_PROFILES, are every other order of the two value sets.

The environments draw these profiles differently (see port_correlation for the correlation between ports):
- correlated environment: a structured profile in CORRELATED_STR_SHARE of blocks, otherwise a shuffled one, each
  profile of a type equally likely (correlated_env_profiles);
- uncorrelated environment: the task's generator, a Gaussian profile around a random best port whose ports are
  then scrambled by random swaps (uncorrelated_env_profiles).
"""
from itertools import permutations, product

import numpy as np

from utils.config import PORTS  # noqa: F401  (re-exported for the figure scripts)

VALUE_SETS = [(80, 46, 15, 10), (80, 46, 46, 15)]
REWARD_PERCENTS = sorted({v for values in VALUE_SETS for v in values})  # 10, 15, 46, 80
STR_PROFILES = np.array([[80, 46, 15, 10], [46, 80, 46, 15], [15, 46, 80, 46], [10, 15, 46, 80]])
_every_order = np.unique([order for values in VALUE_SETS for order in permutations(values)], axis=0)
_is_structured = (_every_order[:, None, :] == STR_PROFILES[None, :, :]).all(axis=2).any(axis=1)
UNSTR_PROFILES = _every_order[~_is_structured]  # sorted rows
UNSTR_EXAMPLES = [1, 12, 18, 31]  # rows of UNSTR_PROFILES used as examples in the figures


CORRELATED_STR_SHARE = 8 / 9  # share of structured blocks in the correlated environment

# uncorrelated environment's generator (task code): reward % at port x when the best port is mu
GAUSS_PEAK, GAUSS_FLOOR, GAUSS_SD = 0.7, 0.1, 0.875


def _gauss_percent(mu, x):
    """Reward % of the generator, rounded as the task does (float32, then truncated to an integer)."""
    return int(np.float32(GAUSS_PEAK * np.exp(-(x - mu) ** 2 / (2 * GAUSS_SD ** 2)) + GAUSS_FLOOR) * 100)


def correlated_env_profiles():
    """(profiles, weights): every structured and shuffled profile, weighted by how often the correlated
    environment uses it."""
    profiles = np.vstack([STR_PROFILES, UNSTR_PROFILES])
    weights = np.r_[np.full(len(STR_PROFILES), CORRELATED_STR_SHARE / len(STR_PROFILES)),
                    np.full(len(UNSTR_PROFILES), (1 - CORRELATED_STR_SHARE) / len(UNSTR_PROFILES))]
    return profiles, weights


def uncorrelated_env_profiles():
    """Every outcome of the uncorrelated environment's generator, all equally likely (one row each): the best
    port mu is 1-4, then for i = 1-4 the port at position i is swapped with a random position (4^4 sequences)."""
    rows = []
    for mu in PORTS:
        for swaps in product(range(len(PORTS)), repeat=len(PORTS)):
            order = list(PORTS)
            for i, j in enumerate(swaps):
                order[j], order[i] = order[i], order[j]
            rows.append([_gauss_percent(mu, x) for x in order])
    return np.array(rows)


def port_correlation(profiles, weights=None):
    """Ports x ports Pearson correlation of the reward % across profiles (rows), weighted by `weights`."""
    cov = np.cov(profiles, rowvar=False, aweights=weights)
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def parse_profile(text):
    """Reward % at ports 1-4 from a rewprobfull string such as '[80 46 15 10]'; None if text is not a string."""
    return [float(x) for x in text.strip('[]').split()] if isinstance(text, str) else None

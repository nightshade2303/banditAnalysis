"""Chance levels drawn as reference lines in the figures."""
from itertools import permutations

import numpy as np

# mean |distance| of a switch from each port when the next port is one of the other three at random
CHANCE_DISTANCE = {1: 2, 2: 4 / 3, 3: 4 / 3, 4: 2}

# reward probabilities (%) of ports 1-4. Every block uses one of two sets of values, in some port order
VALUE_SETS = [(80, 46, 15, 10), (80, 46, 46, 15)]
# the four port orders of the correlated environment (env == 'str')
CORRELATED_PROFILES = [(80, 46, 15, 10), (46, 80, 46, 15), (15, 46, 80, 46), (10, 15, 46, 80)]
# every other port order of the two value sets (env == 'unstr'): 32 profiles
UNCORRELATED_PROFILES = sorted({p for v in VALUE_SETS for p in permutations(v)} - set(CORRELATED_PROFILES))


def chance_from_profiles(profiles):
    """Reward rate of random choices, averaged over the possible profiles with equal weight: each profile's
    chance is the mean of its four port probabilities."""
    return float(np.mean([np.mean(p) for p in profiles]) / 100)


# chance for each set of blocks plotted (keys as in COHORT_STYLE and ENV_STYLE):
# the uncorrelated task draws one of the two value sets, so its chance is the mean over VALUE_SETS (0.4225);
# correlated blocks use the four CORRELATED_PROFILES (0.4225); the uncorrelated blocks given to the
# correlated rats are drawn from the 32 UNCORRELATED_PROFILES (0.4056)
CHANCE_REWARD = {'unstr': chance_from_profiles(VALUE_SETS),
                 'str': chance_from_profiles(CORRELATED_PROFILES),
                 'pure_unstr': chance_from_profiles(VALUE_SETS),
                 'str_in_unstr': chance_from_profiles(CORRELATED_PROFILES),
                 'pure_str': chance_from_profiles(CORRELATED_PROFILES),
                 'unstr_in_str': chance_from_profiles(UNCORRELATED_PROFILES)}

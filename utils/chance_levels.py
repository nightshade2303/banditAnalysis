"""Chance levels drawn as reference lines in the figures."""
import numpy as np

from utils.task_profiles import STR_PROFILES, UNSTR_PROFILES, VALUE_SETS

# mean |distance| of a switch from each port when the next port is one of the other three at random
CHANCE_DISTANCE = {1: 2, 2: 4 / 3, 3: 4 / 3, 4: 2}


def chance_from_profiles(profiles):
    """Reward rate of random choices, averaged over the possible profiles with equal weight: each profile's
    chance is the mean of its four port probabilities (%)."""
    return float(np.mean([np.mean(p) for p in profiles]) / 100)


# chance for each set of blocks plotted (keys as in COHORT_STYLE and ENV_STYLE):
# the uncorrelated task draws one of the two value sets, so its chance is the mean over VALUE_SETS (0.4225);
# structured blocks use the four STR_PROFILES (0.4225); the shuffled blocks given to the correlated rats are
# drawn from the 32 UNSTR_PROFILES, of which 22 come from the lower value set (0.4056)
CHANCE_REWARD = {'unstr': chance_from_profiles(VALUE_SETS),
                 'str': chance_from_profiles(STR_PROFILES),
                 'pure_unstr': chance_from_profiles(VALUE_SETS),
                 'str_in_unstr': chance_from_profiles(STR_PROFILES),
                 'pure_str': chance_from_profiles(STR_PROFILES),
                 'unstr_in_str': chance_from_profiles(UNSTR_PROFILES)}

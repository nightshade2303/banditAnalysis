"""The task's reward profiles (reward % at ports 1-4), from figs/rMAB_figure_1.ipynb: the 4 structured
(correlated) profiles and the 32 shuffled ones (every other ordering of the same two reward sets)."""
from itertools import permutations

import numpy as np

PORTS = [1, 2, 3, 4]
STR_PROFILES = np.array([[80, 46, 15, 10], [46, 80, 46, 15], [15, 46, 80, 46], [10, 15, 46, 80]])
_all = np.unique(list(permutations([80, 46, 15, 10])) + list(permutations([46, 80, 46, 15])), axis=0)
UNSTR_PROFILES = _all[~(_all[:, None, :] == STR_PROFILES[None, :, :]).all(axis=2).any(axis=1)]
UNSTR_EXAMPLES = [1, 12, 18, 31]  # rows of UNSTR_PROFILES used as examples in the figures

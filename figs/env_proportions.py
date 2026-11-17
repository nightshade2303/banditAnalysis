# Schematic: share of structured and shuffled blocks in each environment, drawn as an alluvial plot. Left: block
# type, with example reward profiles (reward % at ports 1-4) next to each block type: the first, the second, dots
# for the rest, and the third (utils/task_profiles.py). Right: environment (correlated, uncorrelated). Each band's
# width is the share of the environment's blocks of that type (correlated: 89% structured, 11% shuffled;
# uncorrelated: 8% structured, 92% shuffled), read off the 0 / 50 / 100% scale on each environment's bar. Bands,
# profiles and their part of the right bar keep the colour of their block type (BLOCK_STYLE); the environment
# labels and the strip right of each environment are in the environment's colour (COHORT_STYLE). Numbers are set
# below, not computed from the data.
# Run cell by cell, or from the repo root: python -m figs.env_proportions

# %% Settings: SHARES[environment][block type], each row sums to 1
SHARES = {'str': {'str': 0.89, 'unstr': 0.11},
          'unstr': {'str': 0.08, 'unstr': 0.92}}
BLOCK_LABELS = {'str': 'Structured blocks', 'unstr': 'Shuffled blocks'}
N_EXAMPLES = 3            # example profiles per block type (dots before the last one)
PROFILE_W = 0.35          # width of the profile column, x units of the alluvial plot (bars 1 apart)
PROFILE_GAP = 0.05        # between the profiles and the left bar

# %% Plot
import matplotlib.pyplot as plt
import pandas as pd

from utils.alluvial import BAR_W, inset_bar_column, plot_alluvial
from utils.plotSettings import BLOCK_STYLE, COHORT_STYLE, save_figure
from utils.task_profiles import PORTS, STR_PROFILES, UNSTR_EXAMPLES, UNSTR_PROFILES

envs = ['str', 'unstr']  # top to bottom, both columns
profiles = {'str': STR_PROFILES[:N_EXAMPLES], 'unstr': UNSTR_PROFILES[UNSTR_EXAMPLES[:N_EXAMPLES]]}
fig, ax = plt.subplots(figsize=(2.7, 3.5))
pos = plot_alluvial(ax, [[SHARES[e][b] for b in envs] for e in envs],
                    left_colors=[BLOCK_STYLE[b]['color'] for b in envs],
                    right_colors=[COHORT_STYLE[e]['color'] for e in envs],
                    left_labels=[BLOCK_LABELS[b] for b in envs],
                    right_labels=[f"{COHORT_STYLE[e]['label']} env." for e in envs],
                    right_ticks=[0, 50, 100], vertical_labels=True, left_pad=PROFILE_W + PROFILE_GAP)
x1 = -BAR_W / 2 - PROFILE_GAP
for j, b in enumerate(envs):  # each block type's profiles beside its left bar
    inset_bar_column(ax, x1 - PROFILE_W, x1, pos['left_top'][j], pos['left_h'][j], profiles[b],
                     BLOCK_STYLE[b]['color'], ylim=(0, 80))
fig.tight_layout()
# one table: shares (environment, block_type, share) and the profiles shown (block_type, example, port_1-4)
shares = pd.DataFrame(SHARES).rename_axis('block_type').melt(ignore_index=False, var_name='environment',
                                                             value_name='share').reset_index()
shown = pd.concat({b: pd.DataFrame(p, columns=[f'port_{i}' for i in PORTS]) for b, p in profiles.items()},
                  names=['block_type', 'example']).reset_index()
source = pd.concat({'share': shares, 'profile': shown}, names=['table']).reset_index(level=0).reset_index(drop=True)
save_figure(fig, 'env_proportions', source_data=source)

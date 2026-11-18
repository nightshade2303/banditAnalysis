# Schematic: share of structured and shuffled blocks in each environment (alluvial plot, utils/alluvial.py).
# Left: each block type as a rounded box, its label above, holding example reward profiles (reward probability
# at ports 1-4, shaded as in the rasters; dots stand for the profiles not shown). Right: the correlated and
# uncorrelated environments; each band's width is the proportion of that environment's blocks of that type, read
# off the 0/0.5/1 ticks. The shares are set below, not computed from the data.
# Run cell by cell, or from the repo root: python -m figs.env_proportions

# %% Settings
SHARES = {'str': {'str': 0.89, 'unstr': 0.11},     # SHARES[environment][block type]; each row sums to 1
          'unstr': {'str': 0.08, 'unstr': 0.92}}
BLOCK_LABELS = {'str': 'Structured blocks', 'unstr': 'Shuffled blocks'}
N_EXAMPLES = 3        # example profiles per block type (dots before the last one)
PROFILE_YTICKS = [0, 0.8]
FIGSIZE = (2.8, 3.9)
# layout, in data units of the alluvial axes (a right bar is 1 high; bar widths as in utils/alluvial.py)
GAP = 0.31            # vertical gap between the structured and shuffled rows
WIDTH = 0.35          # distance between the start of the bands and the right bars
PROFILE_W = 0.45      # width of the profile column
PROFILE_GAP = 0.04    # between the profiles and the start of the bands (the box's right edge)
YAXIS_W = 0.24        # left of the profiles, for their y tick numbers and 'Reward probability'
BOX_PAD_X = 0.03      # box's left edge to 'Reward probability'
BOX_PAD_Y = 0.04      # box edge to the profiles
PORT_ROOM = 0.15      # room at the bottom of the last box for the port numbers and 'Port'
TITLE_PAD = 0.03      # between a box's top and its block-type label
TITLE_ROOM = 0.15     # room above the top box for its label
PROPORTION_PAD = 0.08  # right bars' ticks to the 'Proportion' label (room for the tick numbers)
# font sizes, points
YLABEL_SIZE = 7       # 'Reward probability' and 'Port'
PROPORTION_SIZE = 8.5  # 'Proportion' (was 10)

# %% Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.alluvial import BAR_W, inset_bar_column, plot_alluvial, rounded_box
from utils.plotSettings import BLOCK_STYLE, COHORT_STYLE, save_figure
from utils.raster import RECT_ALPHA, shades
from utils.task_profiles import PORTS, REWARD_PERCENTS, STR_PROFILES, UNSTR_EXAMPLES, UNSTR_PROFILES

ORDER = ['str', 'unstr']  # top to bottom: environments on the right, block types on the left
# example profiles as reward probabilities (the task defines them in %)
profiles = {'str': np.asarray(STR_PROFILES[:N_EXAMPLES]) / 100,
            'unstr': np.asarray(UNSTR_PROFILES[UNSTR_EXAMPLES[:N_EXAMPLES]]) / 100}

fig, ax = plt.subplots(figsize=FIGSIZE)
layout = plot_alluvial(ax, [[SHARES[env][bt] for bt in ORDER] for env in ORDER],
                       left_colors=[BLOCK_STYLE[bt]['color'] for bt in ORDER],
                       right_colors=[COHORT_STYLE[env]['color'] for env in ORDER],
                       left_labels=[], right_labels=[f"{COHORT_STYLE[env]['label']} env." for env in ORDER],
                       right_ticks=[0, 1], right_minor_ticks=[0.2, 0.4, 0.6, 0.8], vertical_labels=True, width=WIDTH,
                       left_pad=PROFILE_W + PROFILE_GAP + YAXIS_W, left_bars=False, right_strip=False, gap=GAP,
                       ticks_side='right', scale_label='Proportion of blocks', right_label_rotation=270,
                       scale_label_pad=PROPORTION_PAD, scale_label_size=PROPORTION_SIZE, ticks_up=True)
ax.set_ylim(ax.get_ylim()[0], -TITLE_ROOM)  # room above the top box for its label (y increases downwards)

# profiles inside each block type's box, with a y axis, and port numbers and 'Port' under the last profile
box_right = -BAR_W / 2  # the bands start here
profiles_right = box_right - PROFILE_GAP
profiles_left = profiles_right - PROFILE_W
box_left = profiles_left - YAXIS_W
for j, block_type in enumerate(ORDER):
    top = layout['left_top'][j] + BOX_PAD_Y
    bottom = layout['left_top'][j] + layout['left_h'][j] - BOX_PAD_Y
    height = bottom - top - PORT_ROOM
    colors = {pct / 100: c for pct, c in shades(BLOCK_STYLE[block_type]['color']).items()}
    inset_bar_column(ax, profiles_left, profiles_right, top, height, profiles[block_type], colors, alpha=RECT_ALPHA,
                     ylim=(0, max(REWARD_PERCENTS) / 100), xticklabels_last=True, peak_label=None,
                     yticks=PROFILE_YTICKS, yticklabels_first=True, fontsize='xx-small')  # x and y numbers alike
    ax.text(box_left + BOX_PAD_X, top + height / 2, 'Reward probability', fontsize=YLABEL_SIZE, rotation=90,
            rotation_mode='anchor', ha='center', va='top')
    ax.text((profiles_left + profiles_right) / 2, bottom, 'Port', ha='center', va='bottom', fontsize=YLABEL_SIZE)
    ax.text(box_left, layout['left_top'][j] - TITLE_PAD, BLOCK_LABELS[block_type], ha='left', va='bottom',
            color=BLOCK_STYLE[block_type]['color'])
fig.tight_layout()

# the boxes go last: their rounded corners need the final aspect of the axes
for j, block_type in enumerate(ORDER):
    top = layout['left_top'][j]
    rounded_box(ax, box_left, box_right, top, top + layout['left_h'][j], BLOCK_STYLE[block_type]['color'])

# source data: one table with the shares (table 'share') and the profiles shown as reward probabilities
# (table 'profile')
shares = (pd.DataFrame(SHARES).rename_axis('block_type')
          .melt(ignore_index=False, var_name='environment', value_name='share').reset_index())
port_columns = [f'port_{p}' for p in PORTS]
shown = pd.concat({bt: pd.DataFrame(p, columns=port_columns) for bt, p in profiles.items()},
                  names=['block_type', 'example']).reset_index()
source = pd.concat({'share': shares, 'profile': shown}, names=['table']).reset_index(level=0).reset_index(drop=True)
save_figure(fig, 'env_proportions', source_data=source)

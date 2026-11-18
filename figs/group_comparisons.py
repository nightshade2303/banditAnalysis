# Groups within the uncorrelated environment: blocked vs unblocked rats, and cued vs uncued (blocked) rats.
# Reward rate, switch rate and switch distance over the first N_TRIALS trials of a block, expert data.
#   a. each comparison twice: first block of each session only, and every block (unblocked sessions are mostly
#      a single block, so their "every block" data are close to their first blocks);
#   b. cued vs uncued from block 1 into block 2 of the same session;
#   c. cued vs uncued: switch rate in the first EARLY_TRIALS trials of block 2, one value per rat (Mann-Whitney U).
# Run cell by cell, or from the repo root: python -m figs.group_comparisons

# %% Settings
COMPARISONS = {'blocked_vs_unblocked': ['blocked', 'unblocked'], 'cued_vs_uncued': ['cued', 'uncued']}
MEASURES = {'reward': ('reward_rate', 'Reward rate'), 'shift_t0': ('switch_rate', 'Switch rate'),
            'switch_dist': ('switch_distance', 'Switch distance (ports)')}  # column: (output name, y label)
BLOCKS = {'first_block': (1, 'First block of the session'), 'every_block': (None, 'Every block')}
EARLY_TRIALS = 10

# %% Load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from utils.config import BLOCK_1_TO_2
from utils.load_data import load_groups
from utils.plot_bars import ERROR_KW
from utils.plot_curves import block_segments_figure, every_block_figure
from utils.plotSettings import GROUP_STYLE, save_figure
from utils.switching import add_switch_distance
from utils.trial_curves import block_trial_matrix, segment_edges, segment_matrix, smooth_curves


def load_with_distance(names, **selection):
    return {g: add_switch_distance(df) for g, df in load_groups(names, **selection).items()}


# %% a: one figure per comparison, measure and block selection
for blocks, (block_group, title) in BLOCKS.items():
    groups = load_with_distance(GROUP_STYLE, block_group=block_group)
    for comparison, names in COMPARISONS.items():
        for col, (measure, ylabel) in MEASURES.items():
            curves = {g: smooth_curves(block_trial_matrix(groups[g], col)) for g in names}
            fig, ax = every_block_figure(curves, GROUP_STYLE, ylabel)
            ax.set_title(title)
            fig.tight_layout()
            save_figure(fig, f'{measure}_{comparison}_{blocks}', source_data=curves)
            plt.close(fig)

# %% b: cued vs uncued from block 1 into block 2 of the same session
groups = load_with_distance(['cued', 'uncued'])
for col, (measure, ylabel) in MEASURES.items():
    curves = {g: smooth_curves(segment_matrix(df, col, BLOCK_1_TO_2), BLOCK_1_TO_2) for g, df in groups.items()}
    fig, ax = block_segments_figure(curves, GROUP_STYLE, BLOCK_1_TO_2, ylabel)
    fig.tight_layout()
    save_figure(fig, f'{measure}_cued_vs_uncued_block1_to_block2', source_data=curves)
    plt.close(fig)

# %% c: cued vs uncued, switch rate in the first EARLY_TRIALS trials of block 2 (not smoothed)
block2_start = segment_edges(BLOCK_1_TO_2)[2]  # column of block 2's first trial in the segment table
early = pd.DataFrame({g: segment_matrix(df, 'shift_t0', BLOCK_1_TO_2)
                      .iloc[:, block2_start:block2_start + EARLY_TRIALS].mean(axis=1)
                      for g, df in groups.items()})
test = mannwhitneyu(early['cued'].dropna(), early['uncued'].dropna())

fig, ax = plt.subplots(figsize=(3.5, 4.5))
for i, g in enumerate(early):
    v = early[g].dropna()
    ax.bar(i, v.mean(), 0.6, yerr=v.sem(), color=GROUP_STYLE[g]['color'], error_kw=ERROR_KW)
    ax.scatter(np.full(len(v), i), v, s=12, color='k', alpha=0.4, zorder=3, linewidths=0)
ax.set_xticks(range(len(early.columns)), [GROUP_STYLE[g]['label'] for g in early])
ax.set(ylabel=f'Switch rate, block 2 trials 1-{EARLY_TRIALS}',
       title=f'Mann-Whitney U = {test.statistic:.0f}, p = {test.pvalue:.2f}')
fig.tight_layout()
save_figure(fig, 'switch_rate_cued_vs_uncued_early_block2', source_data=early)
print(f'cued vs uncued, block 2 trials 1-{EARLY_TRIALS}: U = {test.statistic:.0f}, p = {test.pvalue:.3f}')

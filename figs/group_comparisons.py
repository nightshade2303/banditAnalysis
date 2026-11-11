# Groups within the uncorrelated environment: blocked vs unblocked rats, and cued vs uncued (blocked) rats.
# Reward rate, switch rate and switch distance over the first 100 trials of a block, expert data. Each
# comparison is drawn twice: first block of each session only, and every block. Unblocked sessions are
# mostly a single block, so their "every block" data are close to their first blocks.
# Run cell by cell, or from the repo root: python -m figs.group_comparisons

# %% Settings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from utils.config import BLOCK_1_TO_2, N_TRIALS
from utils.load_data import load_groups
from utils.plot_curves import plot_block_segments, plot_mean_sem
from utils.plotSettings import GROUP_STYLE, save_figure
from utils.switching import add_switch_distance
from utils.trial_curves import block_trial_matrix, segment_edges, segment_matrix, smooth_curves

COMPARISONS = {'blocked_vs_unblocked': ['blocked', 'unblocked'], 'cued_vs_uncued': ['cued', 'uncued']}
METRICS = {'reward': ('reward_rate', 'Reward rate'), 'shift_t0': ('switch_rate', 'Switch rate'),
           'switch_dist': ('switch_distance', 'Switch distance (ports)')}
BLOCKS = {'first_block': (1, 'First block of the session'), 'every_block': (None, 'Every block')}


def plot_comparison(groups, col, ylabel, title):
    """Mean +- SEM across rats of `col` over the first N_TRIALS trials of a block, one line per group."""
    curves = {g: smooth_curves(block_trial_matrix(df, col)) for g, df in groups.items()}
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for g, rats_x_trials in curves.items():
        plot_mean_sem(ax, rats_x_trials, **GROUP_STYLE[g])
    ax.set(xlabel='Trial in block', ylabel=ylabel, title=title, xlim=(1, N_TRIALS), xticks=[1, 25, 50, 75, 100])
    ax.legend(frameon=False, labelcolor='linecolor')
    fig.tight_layout()
    return fig, curves


# %% One figure per comparison, metric and block selection
for blocks, (block_group, title) in BLOCKS.items():
    groups = {g: add_switch_distance(df) for g, df in load_groups(GROUP_STYLE, block_group=block_group).items()}
    for comparison, names in COMPARISONS.items():
        for col, (metric, ylabel) in METRICS.items():
            fig, curves = plot_comparison({g: groups[g] for g in names}, col, ylabel, title)
            save_figure(fig, f'{metric}_{comparison}_{blocks}', source_data=curves)
            plt.close(fig)

# %% Cued vs uncued from block 1 into block 2 of the same session (as plots 2, 3b and 4b)
groups = {g: add_switch_distance(df) for g, df in load_groups(['cued', 'uncued']).items()}
for col, (metric, ylabel) in METRICS.items():
    curves = {g: smooth_curves(segment_matrix(df, col, BLOCK_1_TO_2), BLOCK_1_TO_2) for g, df in groups.items()}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_block_segments(ax, curves, GROUP_STYLE, BLOCK_1_TO_2)
    ax.set(xlabel='Trial in block', ylabel=ylabel)
    ax.legend(frameon=False, labelcolor='linecolor', loc='upper right')
    fig.tight_layout()
    save_figure(fig, f'{metric}_cued_vs_uncued_block1_to_block2', source_data=curves)
    plt.close(fig)

# %% Cued vs uncued: switch rate in the first EARLY trials of block 2, one value per rat (not smoothed)
EARLY = 10
start = segment_edges(BLOCK_1_TO_2)[2]  # first trial of block 2 in the segment table
early = pd.DataFrame({g: segment_matrix(df, 'shift_t0', BLOCK_1_TO_2).iloc[:, start:start + EARLY].mean(axis=1)
                      for g, df in groups.items()})
test = mannwhitneyu(early['cued'].dropna(), early['uncued'].dropna())

fig, ax = plt.subplots(figsize=(3.5, 4.5))
for i, g in enumerate(early):
    v = early[g].dropna()
    ax.bar(i, v.mean(), 0.6, yerr=v.sem(), color=GROUP_STYLE[g]['color'], error_kw=dict(lw=1.5, capsize=0))
    ax.scatter(np.full(len(v), i), v, s=12, color='k', alpha=0.4, zorder=3, linewidths=0)
ax.set_xticks(range(len(early.columns)), [GROUP_STYLE[g]['label'] for g in early])
ax.set(ylabel=f'Switch rate, block 2 trials 1-{EARLY}', title=f'Mann-Whitney U = {test.statistic:.0f}, p = {test.pvalue:.2f}')
fig.tight_layout()
save_figure(fig, 'switch_rate_cued_vs_uncued_early_block2', source_data=early)
print(f'cued vs uncued, block 2 trials 1-{EARLY}: U = {test.statistic:.0f}, p = {test.pvalue:.3f}')

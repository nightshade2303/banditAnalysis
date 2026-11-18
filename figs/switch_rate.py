# Switch rate across trials, uncorrelated (sessdf) vs correlated (hsbdf) rats, expert data. A trial is a switch
# when the next choice in the session is a different port (shift_t0, docs/data_columns.md).
#   a. every block; b. from block 1 into block 2 of the same session;
#   c. per rat, mean over trials 1-100 of block 1 and of block 2 (cohorts compared with a Mann-Whitney U test).
# Run cell by cell, or from the repo root: python -m figs.switch_rate

# %% Settings
COLUMN = 'shift_t0'
YLABEL = 'Switch rate'

# %% Load both cohorts
import matplotlib.pyplot as plt

from utils.config import BLOCK_1_TO_2
from utils.load_data import load_cohorts
from utils.plot_curves import block_means, block_segments_figure, every_block_figure, plot_block_bars
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.trial_curves import block_trial_matrix, segment_matrix, smooth_curves

cohorts = load_cohorts()

# %% a: every block, first trials
curves = {task: smooth_curves(block_trial_matrix(df, COLUMN)) for task, df in cohorts.items()}
fig, ax = every_block_figure(curves, COHORT_STYLE, YLABEL)
fig.tight_layout()
save_figure(fig, 'switch_rate_every_block', source_data=curves)

# %% b: from block 1 into block 2 of the same session
curves = {task: smooth_curves(segment_matrix(df, COLUMN, BLOCK_1_TO_2), BLOCK_1_TO_2) for task, df in cohorts.items()}
fig, ax = block_segments_figure(curves, COHORT_STYLE, BLOCK_1_TO_2, YLABEL)
fig.tight_layout()
save_figure(fig, 'switch_rate_block1_to_block2', source_data=curves)

# %% c: per rat, mean over trials 1-100 of block 1 and of block 2 (sessions of b, not smoothed)
means = {task: block_means(segment_matrix(df, COLUMN, BLOCK_1_TO_2), BLOCK_1_TO_2) for task, df in cohorts.items()}
fig, ax = plt.subplots(figsize=(4.5, 4.5))
tests = plot_block_bars(ax, means, COHORT_STYLE)
ax.set_ylabel(f'{YLABEL} (trials 1-100)')
ax.legend(frameon=False, labelcolor='linecolor', loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
save_figure(fig, 'switch_rate_block1_vs_block2', source_data=means)
print(tests.round(4).to_string(index=False))

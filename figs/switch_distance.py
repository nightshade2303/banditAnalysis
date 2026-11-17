# Plot 4: switch distance (ports between the current and the next choice, on switch trials) across
# trials, uncorrelated (sessdf) vs correlated (hsbdf) rats, expert data.
# Run cell by cell, or from the repo root: python -m figs.switch_distance

# %% Load both cohorts
import matplotlib.pyplot as plt

from utils.config import BLOCK_1_TO_2, N_TRIALS
from utils.load_data import load_cohorts
from utils.plot_curves import block_means, plot_block_bars, plot_block_segments, plot_mean_sem
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.switching import add_switch_distance
from utils.trial_curves import block_trial_matrix, segment_matrix, smooth_curves

cohorts = {task: add_switch_distance(df) for task, df in load_cohorts().items()}

# %% Plot 4a: switch distance over the first 100 trials of every block
curves = {task: smooth_curves(block_trial_matrix(df, 'switch_dist')) for task, df in cohorts.items()}

fig, ax = plt.subplots(figsize=(5, 4.5))
for task, rats_x_trials in curves.items():
    plot_mean_sem(ax, rats_x_trials, **COHORT_STYLE[task])
ax.set(xlabel='Trial in block', ylabel='Switch distance (ports)', xlim=(1, N_TRIALS), xticks=[1, 25, 50, 75, 100])
ax.legend(frameon=False, labelcolor='linecolor')
fig.tight_layout()
save_figure(fig, 'switch_distance_every_block', source_data=curves)

# %% Plot 4b: switch distance from block 1 into block 2 of the same session
curves = {task: smooth_curves(segment_matrix(df, 'switch_dist', BLOCK_1_TO_2), BLOCK_1_TO_2)
          for task, df in cohorts.items()}

fig, ax = plt.subplots(figsize=(9, 4.5))
plot_block_segments(ax, curves, COHORT_STYLE, BLOCK_1_TO_2)
ax.set(xlabel='Trial in block', ylabel='Switch distance (ports)')
ax.legend(frameon=False, labelcolor='linecolor', loc='upper right')
fig.tight_layout()
save_figure(fig, 'switch_distance_block1_to_block2', source_data=curves)

# %% Plot 4c: per rat, mean switch distance over trials 1-100 of block 1 and of block 2 (sessions of plot 4b,
# not smoothed; each trial's mean over the rat's sessions, then the mean over trials). In each block,
# uncorrelated vs correlated rats (Mann-Whitney U, p above the block)
means = {task: block_means(segment_matrix(df, 'switch_dist', BLOCK_1_TO_2), BLOCK_1_TO_2) for task, df in cohorts.items()}
fig, ax = plt.subplots(figsize=(4.5, 4.5))
tests = plot_block_bars(ax, means, COHORT_STYLE)
ax.set_ylabel('Switch distance (trials 1-100)')
ax.legend(frameon=False, labelcolor='linecolor', loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
save_figure(fig, 'switch_distance_block1_vs_block2', source_data=means)
print(tests.round(4).to_string(index=False))

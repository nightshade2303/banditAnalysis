# Plot 3: switch rate across trials, uncorrelated (sessdf) vs correlated (hsbdf) rats, expert data.
# A trial is a switch when the next choice in the session is a different port.
# Run cell by cell, or from the repo root: python -m figs.switch_rate

# %% Load both cohorts
import matplotlib.pyplot as plt

from utils.config import BLOCK_1_TO_2, N_TRIALS
from utils.load_data import load_cohorts
from utils.plot_curves import plot_block_segments, plot_mean_sem
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.trial_curves import block_trial_matrix, segment_matrix, smooth_curves

cohorts = load_cohorts()

# %% Plot 3a: switch rate over the first 100 trials of every block
curves = {task: smooth_curves(block_trial_matrix(df, 'shift_t0')) for task, df in cohorts.items()}

fig, ax = plt.subplots(figsize=(5, 4.5))
for task, rats_x_trials in curves.items():
    plot_mean_sem(ax, rats_x_trials, **COHORT_STYLE[task])
ax.set(xlabel='Trial in block', ylabel='Switch rate', xlim=(1, N_TRIALS), xticks=[1, 25, 50, 75, 100])
ax.legend(frameon=False, labelcolor='linecolor')
fig.tight_layout()
save_figure(fig, 'switch_rate_every_block', source_data=curves)

# %% Plot 3b: switch rate from block 1 into block 2 of the same session
curves = {task: smooth_curves(segment_matrix(df, 'shift_t0', BLOCK_1_TO_2), BLOCK_1_TO_2)
          for task, df in cohorts.items()}

fig, ax = plt.subplots(figsize=(9, 4.5))
plot_block_segments(ax, curves, COHORT_STYLE, BLOCK_1_TO_2)
ax.set(xlabel='Trial in block', ylabel='Switch rate')
ax.legend(frameon=False, labelcolor='linecolor', loc='upper right')
fig.tight_layout()
save_figure(fig, 'switch_rate_block1_to_block2', source_data=curves)

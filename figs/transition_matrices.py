# Choice transition matrices of each cohort, from the first N_TRIALS trials of every block with at least
# N_TRIALS trials (expert data): P(port at trial t+1 | port at trial t). Each rat's matrix is computed first,
# then combined over rats with TM_AGG (mean by default; set it to 'median' in utils/config.py).
#   a. transition probabilities (log colour scale);
#   b. after rewarded minus after unrewarded trials (same colour scale for both cohorts).
# Run cell by cell, or from the repo root: python -m figs.transition_matrices

# %% Settings
VMIN = 1e-2   # lowest probability of the log colour scale in a (lower values get the lowest colour)

# %% Load both cohorts
import matplotlib.pyplot as plt
import numpy as np

from utils.config import N_TRIALS, TM_AGG
from utils.load_data import load_cohorts
from utils.plot_matrices import plot_difference_matrix, plot_transition_matrix, port_matrix_frame
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.transitions import cohort_reward_difference, cohort_transition_matrix

cohorts = load_cohorts(first_trials=N_TRIALS)


def panel_title(task, n_rats):
    return f"{COHORT_STYLE[task]['label']} ({TM_AGG} of {n_rats} rats)"


# %% a: transition matrices
tms = {task: cohort_transition_matrix(df) for task, df in cohorts.items()}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (task, (tm, n_rats)) in zip(axes, tms.items()):
    plot_transition_matrix(ax, tm, panel_title(task, n_rats), VMIN)
fig.tight_layout()
save_figure(fig, 'transition_matrices', source_data={task: port_matrix_frame(tm) for task, (tm, _) in tms.items()})

# %% b: after rewarded minus after unrewarded trials
diffs = {task: cohort_reward_difference(df) for task, df in cohorts.items()}
lim = max(np.nanmax(np.abs(diff)) for diff, _ in diffs.values())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (task, (diff, n_rats)) in zip(axes, diffs.items()):
    plot_difference_matrix(ax, diff, panel_title(task, n_rats), lim)
fig.tight_layout()
save_figure(fig, 'transition_matrices_rewarded_minus_unrewarded',
            source_data={task: port_matrix_frame(diff) for task, (diff, _) in diffs.items()})

# Plot 5: choice transition matrices of each cohort, from the first 100 trials of every block with at
# least 100 trials (expert data). Each rat's matrix is computed first, then combined over rats with
# TM_AGG (mean by default; set it to 'median' in utils/config.py).
# Run cell by cell, or from the repo root: python -m figs.transition_matrices

# %% Load both cohorts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import N_TRIALS, PORTS, TM_AGG
from utils.dfLoading import subset_trials
from utils.load_data import load_cohorts
from utils.plot_matrices import plot_difference_matrix, plot_transition_matrix
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.transitions import cohort_reward_difference, cohort_transition_matrix

cohorts = {task: subset_trials(df, N_TRIALS, head_trials=N_TRIALS) for task, df in load_cohorts().items()}

# %% Plot 5: transition matrices
tms = {task: cohort_transition_matrix(df) for task, df in cohorts.items()}
# vmin = min(np.nanmin(np.where(tm > 0, tm, np.nan)) for tm, _ in tms.values())  # same log colour scale for both cohorts
vmin = 1e-2

matrices = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (task, (tm, n_rats)) in zip(axes, tms.items()):
    matrices[task] = pd.DataFrame(tm, index=pd.Index(PORTS, name='port at t'), columns=PORTS)
    plot_transition_matrix(ax, tm, f"{COHORT_STYLE[task]['label']} ({TM_AGG} of {n_rats} rats)", vmin)
fig.tight_layout()
save_figure(fig, 'transition_matrices', source_data=matrices)

# %% Plot 5, after rewarded minus after unrewarded trials (notebook cell 21)
diffs = {task: cohort_reward_difference(df) for task, df in cohorts.items()}
lim = max(np.nanmax(np.abs(diff)) for diff, _ in diffs.values())  # same colour scale for both cohorts

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (task, (diff, n_rats)) in zip(axes, diffs.items()):
    plot_difference_matrix(ax, diff, f"{COHORT_STYLE[task]['label']} ({TM_AGG} of {n_rats} rats)", lim)
fig.tight_layout()
save_figure(fig, 'transition_matrices_rewarded_minus_unrewarded',
            source_data={task: pd.DataFrame(diff, index=pd.Index(PORTS, name='port at t'), columns=PORTS)
                         for task, (diff, _) in diffs.items()})

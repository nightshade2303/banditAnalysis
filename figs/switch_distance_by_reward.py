# Plot 8: switch distance after rewarded and unrewarded trials, uncorrelated (sessdf) vs correlated
# (hsbdf) rats. First block of each session only (its first 100 trials, blocks with at least 100
# trials), expert data. Dashed lines: chance distance for the ports the rats switched from.
# Run cell by cell, or from the repo root: python -m figs.switch_distance_by_reward

# %% Load both cohorts, first block only
import matplotlib.pyplot as plt

from utils.config import N_TRIALS
from utils.dfLoading import subset_trials
from utils.load_data import load_cohorts
from utils.plot_bars import plot_outcome_bars
from utils.plot_curves import plot_mean_sem
from utils.plotSettings import COHORT_STYLE, REWARD_STYLE, save_figure
from utils.switching import add_switch_distance, by_outcome, switch_distance_by_reward
from utils.trial_curves import block_trial_matrix, smooth_curves

cohorts = {task: add_switch_distance(subset_trials(df, N_TRIALS, head_trials=N_TRIALS))
           for task, df in load_cohorts(block_group=1).items()}

# %% Plot 8: mean switch distance per rat after rewarded / unrewarded trials (notebook cell 113)
stats = {task: switch_distance_by_reward(df) for task, df in cohorts.items()}
distance = {task: s['distance'].unstack() for task, s in stats.items()}
chance = {task: s['chance'].unstack() for task, s in stats.items()}

fig, ax = plt.subplots(figsize=(4, 4.5))
plot_outcome_bars(ax, distance, COHORT_STYLE, chance=chance)
ax.set(ylabel='Switch distance (ports)', ylim=(1, None))
ax.legend(frameon=False, labelcolor='linecolor', ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
fig.tight_layout()
save_figure(fig, 'switch_distance_by_reward', source_data=stats)

# %% Plot 8, trial by trial: switch distance after each outcome at each trial of block 1 (cells 120-122)
curves = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, reward in zip(axes, [1, 0]):
    for task, df in cohorts.items():
        curves[f'{task}, reward {reward}'] = c = smooth_curves(block_trial_matrix(by_outcome(df, 'switch_dist', reward), 'switch_dist'))
        plot_mean_sem(ax, c, **COHORT_STYLE[task])
    ax.set(title=f"After {REWARD_STYLE[reward]['label'].lower()} trials", xlabel='Trial in block 1',
           xlim=(1, N_TRIALS), xticks=[1, 25, 50, 75, 100])
axes[0].set_ylabel('Switch distance (ports)')
axes[0].legend(frameon=False, labelcolor='linecolor')
fig.tight_layout()
save_figure(fig, 'switch_distance_by_reward_trialwise', source_data=curves)

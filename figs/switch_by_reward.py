# Probability of switching after rewarded and unrewarded trials, uncorrelated (sessdf) vs correlated (hsbdf)
# rats. First block of each session only (blocks with at least N_TRIALS trials, their first N_TRIALS), expert
# data.
#   a. one value per rat: P(switch | outcome);
#   b. trial by trial: P(switch | outcome) at each trial of block 1.
# Run cell by cell, or from the repo root: python -m figs.switch_by_reward

# %% Settings
COLUMN = 'shift_t0'
YLABEL = 'P(switch)'

# %% Load both cohorts, first block only
import matplotlib.pyplot as plt

from utils.config import N_TRIALS, TRIAL_XLABEL
from utils.load_data import load_cohorts
from utils.plot_bars import plot_outcome_bars
from utils.plot_curves import format_trial_axis, plot_mean_sem
from utils.plotSettings import COHORT_STYLE, REWARD_STYLE, save_figure
from utils.switching import by_outcome, switch_rate_by_reward
from utils.trial_curves import block_trial_matrix, smooth_curves

cohorts = load_cohorts(block_group=1, first_trials=N_TRIALS)

# %% a: P(switch | outcome), one value per rat
p_switch = {task: switch_rate_by_reward(df) for task, df in cohorts.items()}

fig, ax = plt.subplots(figsize=(4, 4.5))
plot_outcome_bars(ax, p_switch, COHORT_STYLE)
ax.set_ylabel(YLABEL)
ax.legend(frameon=False, labelcolor='linecolor', ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
fig.tight_layout()
save_figure(fig, 'switch_prob_by_reward', source_data=p_switch)

# %% b: trial by trial, one panel per outcome
curves = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, reward in zip(axes, [1, 0]):
    for task, df in cohorts.items():
        trials = block_trial_matrix(by_outcome(df, COLUMN, reward), COLUMN)
        curves[f'{task}, reward {reward}'] = smooth_curves(trials)
        plot_mean_sem(ax, curves[f'{task}, reward {reward}'], **COHORT_STYLE[task])
    format_trial_axis(ax)
    ax.set(title=f"After {REWARD_STYLE[reward]['label'].lower()} trials", xlabel=f'{TRIAL_XLABEL} 1')
axes[0].set_ylabel(YLABEL)
axes[0].legend(frameon=False, labelcolor='linecolor')
fig.tight_layout()
save_figure(fig, 'switch_prob_by_reward_trialwise', source_data=curves)

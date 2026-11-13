# Plots 1-2: reward rate across trials, uncorrelated (sessdf) vs correlated (hsbdf) rats, expert data.
# Run cell by cell, or from the repo root: python -m figs.reward_rate

# %% Load both cohorts
import matplotlib.pyplot as plt

from utils.chance_levels import CHANCE_REWARD
from utils.config import BLOCK_1_TO_2, N_TRIALS
from utils.load_data import load_cohorts
from utils.plot_curves import plot_block_segments, plot_mean_sem
from utils.plotSettings import COHORT_STYLE, ENV_STYLE, save_figure
from utils.trial_curves import block_trial_matrix, segment_matrix, smooth_curves

cohorts = load_cohorts()

# %% Plot 1: reward rate over the first 100 trials of every block
curves = {task: smooth_curves(block_trial_matrix(df, 'reward')) for task, df in cohorts.items()}

fig, ax = plt.subplots(figsize=(5, 4.5))
for task, rats_x_trials in curves.items():
    plot_mean_sem(ax, rats_x_trials, **COHORT_STYLE[task])
    ax.axhline(CHANCE_REWARD[task], color=COHORT_STYLE[task]['color'], linestyle='--', linewidth=0.75)
ax.set(xlabel='Trial in block', ylabel='Reward rate', xlim=(1, N_TRIALS), xticks=[1, 25, 50, 75, 100])
ax.legend(frameon=False, labelcolor='linecolor')
fig.tight_layout()
save_figure(fig, 'reward_rate_every_block', source_data=curves)

# %% Plot 1 split by the block's reward profile (env): blocks with one of the four correlated profiles vs the
# rest, within each cohort (notebook cells 74-75). Every block with at least 100 trials, its first 100 trials.
ENV_BLOCKS = {'pure_unstr': ('unstr', 'unstr'), 'str_in_unstr': ('unstr', 'str'),
              'pure_str': ('str', 'str'), 'unstr_in_str': ('str', 'unstr')}  # (cohort, env)
PANELS = {'uncorrelated_vs_correlated': [('cohort', 'unstr'), ('cohort', 'str')],
          'uncorrelated_cohort': [('env', 'pure_unstr'), ('env', 'str_in_unstr')],
          'correlated_cohort': [('env', 'pure_str'), ('env', 'unstr_in_str')]}
blocks = {k: cohorts[task][cohorts[task].env == env] for k, (task, env) in ENV_BLOCKS.items()}
style = {**{('cohort', t): s for t, s in COHORT_STYLE.items()}, **{('env', k): s for k, s in ENV_STYLE.items()}}
data = {**{('cohort', t): df for t, df in cohorts.items()}, **{('env', k): df for k, df in blocks.items()}}
curves = {key: smooth_curves(block_trial_matrix(df, 'reward')) for key, df in data.items()}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, keys in zip(axes, PANELS.values()):
    for key in keys:
        plot_mean_sem(ax, curves[key], **style[key])
        ax.axhline(CHANCE_REWARD[key[1]], color=style[key]['color'],
                   linestyle='--', linewidth=0.75)
    ax.set(xlabel='Trial in block', xlim=(1, N_TRIALS), xticks=[1, 25, 50, 75, 100])
    ax.legend(frameon=False, labelcolor='linecolor', loc='center right')
axes[0].set_ylabel('Reward rate')
fig.tight_layout()
save_figure(fig, 'reward_rate_every_block_by_env', source_data={f'{a}_{b}': c for (a, b), c in curves.items()})
for key, c in curves.items():
    print(key, 'rats:', len(c), 'blocks:', data[key].groupby(['animal', 'block_num']).ngroups)

# %% Plot 2: reward rate from block 1 into block 2 of the same session
curves = {task: smooth_curves(segment_matrix(df, 'reward', BLOCK_1_TO_2), BLOCK_1_TO_2)
          for task, df in cohorts.items()}

fig, ax = plt.subplots(figsize=(9, 4.5))
plot_block_segments(ax, curves, COHORT_STYLE, BLOCK_1_TO_2)
for task in curves:
    ax.axhline(CHANCE_REWARD[task], color=COHORT_STYLE[task]['color'], linestyle='--', linewidth=0.75)
ax.set(xlabel='Trial in block', ylabel='Reward rate')
ax.legend(frameon=False, labelcolor='linecolor', loc='lower right')
fig.tight_layout()
save_figure(fig, 'reward_rate_block1_to_block2', source_data=curves)

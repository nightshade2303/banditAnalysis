# Week-wise learning: reward rate, switch rate and switch distance over the first N_TRIALS trials of the first
# block of each session, for each week of training (weeks 1 to MAX_WEEK), one figure per cohort. All rats and
# all days are used, so the number of rats and sessions changes from week to week (last figure).
# Run cell by cell, or from the repo root: python -m figs.weekwise

# %% Settings
MEASURES = {'reward': 'Reward rate', 'shift_t0': 'Switch rate', 'switch_dist': 'Switch distance (ports)'}
COUNT_BAR_WIDTH = 0.4

# %% Load both cohorts, first block of each session, weeks 1 to MAX_WEEK
import matplotlib.pyplot as plt
import pandas as pd

from utils.config import MAX_WEEK, N_TRIALS
from utils.dfLoading import subset_trials
from utils.load_data import load_cohorts
from utils.plot_curves import format_trial_axis, plot_mean_sem
from utils.plotSettings import COHORT_STYLE, save_figure, week_colors
from utils.switching import add_switch_distance
from utils.trial_curves import block_trial_matrix, smooth_curves

cohorts = {task: add_switch_distance(df[df.week_task <= MAX_WEEK])
           for task, df in load_cohorts(after_day=0, block_group=1, min_blocks=None).items()}
weeks = range(1, MAX_WEEK + 1)
colors = week_colors(MAX_WEEK)

# %% Reward rate, switch rate and switch distance for each week, one figure per cohort
for task, df in cohorts.items():
    curves = {(col, week): smooth_curves(block_trial_matrix(df[df.week_task == week], col))
              for col in MEASURES for week in weeks}
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(14, 4.5))
    for ax, (col, ylabel) in zip(axes, MEASURES.items()):
        for week in weeks:
            plot_mean_sem(ax, curves[col, week], colors[week - 1], f'Week {week}')
        format_trial_axis(ax)
        ax.set_ylabel(ylabel)
    axes[-1].legend(frameon=False, labelcolor='linecolor', loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(COHORT_STYLE[task]['label'])
    fig.tight_layout()
    save_figure(fig, f'weekwise_{task}', source_data=pd.concat(curves, names=['metric', 'week']))

# %% Number of sessions and rats behind each week (blocks of at least N_TRIALS trials, as in the curves)
counts = {}
for task, df in cohorts.items():
    blocks = subset_trials(df, N_TRIALS, head_trials=N_TRIALS)
    sessions = blocks.drop_duplicates(['animal', 'session'])
    counts[task] = pd.DataFrame({'sessions': sessions.groupby('week_task').size(),
                                 'rats': blocks.groupby('week_task').animal.nunique()})
counts = pd.concat(counts, names=['cohort'])

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, col in zip(axes, ['sessions', 'rats']):
    for i, task in enumerate(cohorts):
        n = counts.loc[task, col]
        ax.bar(n.index + (i - 0.5) * COUNT_BAR_WIDTH, n, COUNT_BAR_WIDTH, color=COHORT_STYLE[task]['color'],
               label=COHORT_STYLE[task]['label'])
    ax.set(xlabel='Week of training', ylabel=f'Number of {col}', xticks=list(weeks))
axes[0].legend(frameon=False, labelcolor='linecolor')
fig.tight_layout()
save_figure(fig, 'weekwise_counts', source_data=counts)

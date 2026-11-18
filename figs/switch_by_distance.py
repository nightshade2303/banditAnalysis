# Probability of the next choice being at each distance (ports) from the current one, uncorrelated (sessdf)
# vs correlated (hsbdf) rats. First block of each session only (blocks with at least N_TRIALS trials, their
# first N_TRIALS), expert data. Dashed lines: chance, i.e. the rat's own switch rate with the next port drawn
# at random from the other three (utils/switching.py switch_distance_probs).
#   a. after rewarded and unrewarded trials (distances 1-3);
#   b. whatever the outcome; distance 0 is P(stay), with no chance line (chance keeps the rat's stay rate).
# Log y axis: the larger distances are rare.
# Run cell by cell, or from the repo root: python -m figs.switch_by_distance

# %% Settings
XLABEL = 'Switch distance'
YLABEL = 'P (switch)'

# %% Load both cohorts, first block only
import matplotlib.pyplot as plt

from utils.config import N_TRIALS
from utils.load_data import load_cohorts
from utils.plot_bars import plot_distance_bars, plot_distance_bars_any_outcome
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.switching import switch_distance_probs

cohorts = load_cohorts(block_group=1, first_trials=N_TRIALS)

# %% a: P(switch to distance d | outcome)
probs = {task: switch_distance_probs(df) for task, df in cohorts.items()}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (task, p) in zip(axes, probs.items()):
    plot_distance_bars(ax, p)
    ax.set(title=f"{COHORT_STYLE[task]['label']} (n = {p.animal.nunique()})", xlabel=XLABEL)
    ax.set_yscale('log', nonpositive='clip')
axes[0].set_ylabel(YLABEL)
axes[0].legend(frameon=False)
fig.tight_layout()
save_figure(fig, 'switch_prob_by_distance', source_data=probs)

# %% b: whatever the outcome; each panel has its own y axis and legend
probs = {task: switch_distance_probs(df, by_outcome=False, include_stay=True) for task, df in cohorts.items()}

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (task, p) in zip(axes, probs.items()):
    plot_distance_bars_any_outcome(ax, p, COHORT_STYLE[task]['color'])
    ax.set(title=COHORT_STYLE[task]['label'], xlabel=XLABEL, ylabel=YLABEL)
    ax.set_yscale('log', nonpositive='clip')
    ax.legend(frameon=False)
fig.tight_layout()
save_figure(fig, 'switch_prob_by_distance_any_outcome', source_data=probs)

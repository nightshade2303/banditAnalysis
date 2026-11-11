# Plot 7: probability of switching to a port at each distance after rewarded and unrewarded trials,
# uncorrelated (sessdf) vs correlated (hsb_data) rats. First block of each session only (its first 100
# trials, blocks with at least 100 trials), expert data. Dashed lines: chance, i.e. the rat's switch rate
# with the next port drawn at random from the other three.
# Run cell by cell, or from the repo root: python -m figs.switch_by_distance

# %% Load both cohorts, first block only
import matplotlib.pyplot as plt

from utils.config import N_TRIALS
from utils.dfLoading import subset_trials
from utils.load_data import load_cohorts
from utils.plot_bars import plot_distance_bars, plot_distance_bars_any_outcome
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.switching import switch_distance_probs

cohorts = {task: subset_trials(df, N_TRIALS, head_trials=N_TRIALS)
           for task, df in load_cohorts(block_group=1).items()}

# %% Plot 7: P(switch to distance d | outcome) (notebook cell 137)
probs = {task: switch_distance_probs(df) for task, df in cohorts.items()}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (task, p) in zip(axes, probs.items()):
    plot_distance_bars(ax, p)
    ax.set(title=f"{COHORT_STYLE[task]['label']} (n = {p.animal.nunique()})", xlabel='Switch distance (ports)')
    ax.set_yscale('log', nonpositive='clip')  # small probabilities at the larger distances
axes[0].set_ylabel('P(switch to that distance)')
axes[0].legend(frameon=False)
fig.tight_layout()
save_figure(fig, 'switch_prob_by_distance', source_data=probs)

# %% Plot 7, whatever the outcome; distance 0 is P(stay)
probs = {task: switch_distance_probs(df, by_outcome=False, include_stay=True) for task, df in cohorts.items()}

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
for ax, (task, p) in zip(axes, probs.items()):
    plot_distance_bars_any_outcome(ax, p, COHORT_STYLE[task]['color'])
    ax.set(title=COHORT_STYLE[task]['label'], xlabel='Distance to the next choice (ports)')
    ax.set_yscale('log', nonpositive='clip')
axes[0].set_ylabel('P(next choice at that distance)')
axes[0].legend(frameon=False)
fig.tight_layout()
save_figure(fig, 'switch_prob_by_distance_any_outcome', source_data=probs)

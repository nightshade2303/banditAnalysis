# Reward rate, switch rate and switch distance over the first 300 trials of each session, uncorrelated (sessdf)
# vs correlated (hsbdf) rats, expert data.
#
# Per rat: in each session, the mean over its first N_FIRST trials in trial order (across block switches);
# sessions shorter than N_FIRST trials are dropped. The rat's value is the mean over its sessions. Trials
# without a logged reward are left out of the reward rate, stay trials out of the switch distance.
# Box plot per cohort (cohort colour), each rat a dark dot. Cohorts compared with an unpaired permutation test
# on the difference of means (stats/permutation_tests.py, a port of permTest_2samples.m).
# Run cell by cell, or from the repo root: python -m figs.cohort_first_300

# %% Settings
N_FIRST = 300                 # trials from the start of each session
N_PERMS = 10_000
MEASURES = {'reward': 'Reward rate', 'shift_t0': 'Switch rate', 'switch_dist': 'Switch distance'}
Y_MIN = {'shift_t0': 0, 'switch_dist': 1}
COHORT_ORDER = ['unstr', 'str']
DOT_COLOR = '0.15'
DOT_SIZE = 25                 # marker area, pt^2
BOX_ALPHA = 1                 # opacity of the box fill (0 transparent, 1 opaque)
JITTER = 0.05                 # half-width of the dots' horizontal jitter, x units (boxes are 0.5 wide)

# %% Load both cohorts; per-rat means over the first N_FIRST trials of each session
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from stats.permutation_tests import perm_test_2samples
from utils.load_data import load_cohorts
from utils.plot_curves import p_text
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.switching import add_switch_distance


def first_trials_per_rat(df, n_first, measures):
    """Rats x measures: each session's mean over its first n_first trials, averaged over the rat's sessions.
    Sessions with fewer than n_first trials are dropped. Adds the number of sessions used per rat."""
    df = df.sort_values(['animal', 'session', 'trial'])
    session_length = df.groupby(['animal', 'session']).trial.transform('size')
    first = df[(session_length >= n_first) & (df.groupby(['animal', 'session']).cumcount() < n_first)]
    per_session = first.groupby(['animal', 'session'])[measures].mean()
    per_rat = per_session.groupby('animal').mean()
    per_rat['sessions'] = per_session.groupby('animal').size()
    return per_rat


cohorts = {task: add_switch_distance(df) for task, df in load_cohorts().items()}
per_rat = pd.concat({task: first_trials_per_rat(df, N_FIRST, list(MEASURES)) for task, df in cohorts.items()},
                    names=['cohort'])
for task in COHORT_ORDER:
    print(f'{task}: {len(per_rat.loc[task])} rats, {per_rat.loc[task].sessions.sum()} sessions '
          f'with >= {N_FIRST} trials')


# %% Box plots and permutation tests
def cohort_boxplot(ax, values, rng, alpha=1):
    """One box per cohort in its colour (no outline, fill opacity alpha), each rat a dot.
    values: {cohort: per-rat values}."""
    boxes = ax.boxplot(list(values.values()), positions=range(len(values)), widths=0.5, showfliers=False,
                       patch_artist=True, medianprops=dict(color='k'), boxprops=dict(edgecolor='none'),
                       whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for box, cohort in zip(boxes['boxes'], values):
        box.set_facecolor(to_rgba(COHORT_STYLE[cohort]['color'], alpha))
    for x, v in enumerate(values.values()):
        ax.scatter(x + rng.uniform(-JITTER, JITTER, len(v)), v, s=DOT_SIZE, color=DOT_COLOR, zorder=3,
                   linewidths=0)
    ax.set_xticks(range(len(values)), [COHORT_STYLE[c]['label'] for c in values], rotation=30, ha='right',
                  rotation_mode='anchor')
    for label, cohort in zip(ax.get_xticklabels(), values):
        label.set_color(COHORT_STYLE[cohort]['color'])
    ax.set_xlim(-0.6, len(values) - 0.4)


rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, len(MEASURES), figsize=(9, 4.5))
tests = []
for ax, (measure, label) in zip(axes, MEASURES.items()):
    values = {c: per_rat.loc[c, measure].dropna() for c in COHORT_ORDER}
    cohort_boxplot(ax, values, rng, alpha=BOX_ALPHA)
    ax.set_ylabel(label)
    if measure in Y_MIN:
        ax.set_ylim(bottom=Y_MIN[measure])

    p, diff, _ = perm_test_2samples(*values.values(), n_perms=N_PERMS)
    ax.text(0.5, 1.0, p_text(p) if p > 0 else f'p < {1 / N_PERMS:g}', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9)
    tests.append({'measure': measure, **{f'mean_{c}': v.mean() for c, v in values.items()}, 'diff': diff,
                  'p': p, **{f'n_{c}': len(v) for c, v in values.items()}, 'n_perms': N_PERMS})

fig.suptitle(f'First {N_FIRST} trials of each session', fontsize='medium')
fig.tight_layout()
# save_figure(fig, f'cohort_first_{N_FIRST}_trials', source_data=per_rat)
tests = pd.DataFrame(tests)
# tests.to_csv(f'figs/intermediate/cohort_first_{N_FIRST}_trials_tests.csv', index=False)
print(tests.round(4).to_string(index=False))

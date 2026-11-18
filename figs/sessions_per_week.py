# Sessions per week for each rat, for the three datasets: uncorrelated unblocked (UU), uncorrelated blocked (UB)
# and correlated (structured) blocked (SB). One bar per rat, coloured by dataset and grouped by dataset; over
# each group, horizontal lines give the mean (solid), median (dashed) and, for expert data, minimum (dotted)
# across its rats. Sessions are counted after cleaning, whatever the number or length of their blocks. A rat's
# sessions per week = its sessions / the weeks in which it ran at least one session, so weeks with no session
# at all (e.g. after an implant) do not count.
#   a. training (weeks 1 to MAX_WEEK, all rats), week by week;
#   b. sessions per week pooled over training, and over expert data (rats of the expert plots);
#   c. expert data, all rats: total sessions; rats left out of the expert plots are hatched.
# Run cell by cell, or from the repo root: python -m figs.sessions_per_week

# %% Settings
BAR_WIDTH = 0.7
GROUP_GAP = 2                # empty bar slots between datasets
SESSIONS_YLIM = 28           # y limit of the per-week plots (a, b); the summary text sits at SESSIONS_TEXT_Y
SESSIONS_TEXT_Y = 23
TOTAL_YLIM = 520             # same for the total-sessions plot (c)
TOTAL_TEXT_Y = 440
SUMMARY_LINES = {'mean': '-', 'median': '--', 'min': ':'}   # line style of each summary statistic

# %% Load the three datasets
import matplotlib.pyplot as plt
import pandas as pd

from utils.config import MAX_WEEK
from utils.load_data import load_cohorts, load_groups
from utils.plotSettings import COHORT_STYLE, GROUP_STYLE, save_figure

DATASET_COLORS = {'UU': GROUP_STYLE['unblocked']['color'], 'UB': GROUP_STYLE['blocked']['color'],
                  'SB': COHORT_STYLE['str']['color']}


def load_datasets(**selection):
    """{'UU', 'UB', 'SB': trials}. selection: arguments of load_cohorts / load_groups (after_day, min_blocks);
    their defaults (expert data, rats of the expert plots) if not given."""
    cohorts = load_cohorts(**selection)
    unblocked = load_groups(['unblocked'], **selection)['unblocked']
    return {'UU': unblocked, 'UB': cohorts['unstr'], 'SB': cohorts['str']}


def sessions_per_week(df):
    """Per rat: sessions, weeks with at least one session, and sessions per week, sorted by sessions per week."""
    per_rat = df.drop_duplicates(['animal', 'session']).groupby('animal').agg(
        sessions=('session', 'size'), weeks=('week_task', 'nunique'))
    return per_rat.assign(per_week=per_rat.sessions / per_rat.weeks).sort_values('per_week')


def summary_lines(ax, x, values, stats, text_y, text_fmt, fontsize=8):
    """Over the bars at x (one dataset), a line for each statistic in `stats` (names of SUMMARY_LINES) of
    `values` (NaN left out), and their values written above at text_y as text_fmt({name: value})."""
    summary = {name: getattr(values, name)() for name in stats}
    for name, y in summary.items():
        ax.hlines(y, x[0] - BAR_WIDTH / 2 - 0.05, x[-1] + BAR_WIDTH / 2 + 0.05, color='k', ls=SUMMARY_LINES[name],
                  lw=1.2)
    ax.text((x[0] + x[-1]) / 2, text_y, text_fmt(summary), ha='center', va='bottom', fontsize=fontsize)


def stats_text(decimals):
    """text_fmt for summary_lines: one 'name = value' line per statistic."""
    return lambda summary: '\n'.join(f'{name} = {y:.{decimals}f}' for name, y in summary.items())


def label_rats(ax, x, names, fontsize=8):
    """Rat names under the bars at x, slanted."""
    ax.set_xticks(x, names, rotation=70, ha='right', fontsize=fontsize)


training = {k: df[df.week_task <= MAX_WEEK] for k, df in load_datasets(after_day=0, min_blocks=None).items()}
expert = load_datasets()

# %% a: training, week by week: sessions each rat ran in each week (rats in the same place in every panel)
counts = {k: df.drop_duplicates(['animal', 'session']).groupby(['week_task', 'animal']).size().unstack()
          for k, df in training.items()}  # weeks x rats; NaN = no session that week
fig, axes = plt.subplots(MAX_WEEK // 2, 2, figsize=(18, 2.6 * MAX_WEEK // 2), sharex=True, sharey=True)
for week, ax in zip(range(1, MAX_WEEK + 1), axes.flat):
    x0, ticks, names = 0, [], []
    for k, weeks_x_rats in counts.items():
        rats = sorted(weeks_x_rats.columns)
        this_week = weeks_x_rats.reindex(index=[week], columns=rats).iloc[0]
        x = list(range(x0, x0 + len(rats)))
        ax.bar(x, this_week.fillna(0), BAR_WIDTH, color=DATASET_COLORS[k], label=k)
        n_rats = this_week.notna().sum()
        summary_lines(ax, x, this_week, ['mean', 'median'], SESSIONS_TEXT_Y, fontsize=7,
                      text_fmt=lambda s: f"n = {n_rats}, mean = {s['mean']:.1f}, median = {s['median']:.1f}")
        ticks += x
        names += rats
        x0 += len(rats) + GROUP_GAP
    ax.set(title=f'Week {week}', ylim=(0, SESSIONS_YLIM), xlim=(-1, x0))
for ax in axes[-1]:
    label_rats(ax, ticks, names, fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel('Sessions')
axes[0, -1].legend(frameon=False, labelcolor='linecolor', loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
save_figure(fig, 'sessions_by_week_training', source_data=pd.concat(counts, names=['dataset']))

# %% b: sessions per week pooled over training, and over expert data
PERIODS = {'training': (training, f'Weeks 1-{MAX_WEEK}', ['mean', 'median']),
           'expert': (expert, 'Expert data', ['mean', 'median', 'min'])}
for period, (datasets, title, stats) in PERIODS.items():
    table = {k: sessions_per_week(df) for k, df in datasets.items()}
    fig, ax = plt.subplots(figsize=(16, 4.5))
    x0, ticks, names = 0, [], []
    for k, per_rat in table.items():
        x = list(range(x0, x0 + len(per_rat)))
        ax.bar(x, per_rat.per_week, BAR_WIDTH, color=DATASET_COLORS[k], label=k)
        summary_lines(ax, x, per_rat.per_week, stats, SESSIONS_TEXT_Y, stats_text(1))
        ticks += x
        names += list(per_rat.index)
        x0 += len(per_rat) + GROUP_GAP
    label_rats(ax, ticks, names)
    ax.set(ylabel='Sessions per week', title=title, xlim=(-1, x0), ylim=(0, SESSIONS_YLIM))
    ax.legend(frameon=False, labelcolor='linecolor', loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save_figure(fig, f'sessions_per_week_{period}', source_data=table)
    for k, per_rat in table.items():
        print(period, k, len(per_rat), per_rat.per_week.describe()[['mean', '50%', 'min', 'max']].round(2).to_dict())

# %% c: expert data, all rats: total sessions after day EXPERT_AFTER_DAY. Rats left out of the expert plots
# (no expert data, or, if MIN_BLOCKS is set, MIN_BLOCKS or fewer blocks of at least N_TRIALS trials) are hatched;
# rats with no expert session show 0.
# Mean, median and minimum are over all rats of the dataset.
expert_all_rats = load_datasets(min_blocks=None)
fig, ax = plt.subplots(figsize=(16, 4.5))
x0, ticks, names, table = 0, [], [], {}
for k, df in expert_all_rats.items():
    sessions = (df.drop_duplicates(['animal', 'session']).groupby('animal').size()
                .reindex(sorted(training[k].animal.unique()), fill_value=0).sort_values())
    in_plots = sessions.index.isin(expert[k].animal.unique())
    table[k] = sessions.to_frame('sessions').assign(in_expert_plots=in_plots)
    x = list(range(x0, x0 + len(sessions)))
    for xi, n, kept in zip(x, sessions, in_plots):  # hatched, unfilled bar: rat not in the expert plots
        ax.bar(xi, n, BAR_WIDTH, color=DATASET_COLORS[k] if kept else 'none', edgecolor=DATASET_COLORS[k],
               hatch=None if kept else '///')
    summary_lines(ax, x, sessions, ['mean', 'median', 'min'], TOTAL_TEXT_Y, stats_text(0))
    ax.bar(0, 0, color=DATASET_COLORS[k], label=k)  # legend entry
    ticks += x
    names += list(sessions.index)
    x0 += len(sessions) + GROUP_GAP
ax.bar(0, 0, color='none', edgecolor='grey', hatch='///', label='not in expert plots')
label_rats(ax, ticks, names)
ax.set(ylabel='Sessions', title='Expert data: total sessions per rat', xlim=(-1, x0), ylim=(0, TOTAL_YLIM))
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
save_figure(fig, 'sessions_total_expert', source_data=table)
for k, t in table.items():
    print('expert total', k, len(t), t.sessions.describe()[['mean', '50%', 'min', 'max']].round(1).to_dict())

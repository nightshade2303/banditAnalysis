# Sessions per week for each rat, pooled over a training period, for the three datasets: uncorrelated
# unblocked (UU), uncorrelated blocked (UB) and correlated (structured) blocked (SB). One bar per rat,
# coloured by dataset; for each dataset, horizontal lines over its bars give the mean and median across its
# rats (and the minimum for expert data). Sessions are counted after cleaning, whatever the number or length
# of their blocks. A rat's sessions per week = its sessions / the weeks in which it ran at least one session,
# so weeks with no sessions at all (e.g. after an implant) do not count.
# Run cell by cell, or from the repo root: python -m figs.sessions_per_week

# %% Load the three datasets: training (weeks 1 to MAX_WEEK, all rats) and expert data (rats of the expert plots)
import matplotlib.pyplot as plt
import pandas as pd

from utils.config import MAX_WEEK
from utils.load_data import load_cohorts, load_groups
from utils.plotSettings import COHORT_STYLE, GROUP_STYLE, save_figure

DATASETS = {'UU': GROUP_STYLE['unblocked']['color'], 'UB': GROUP_STYLE['blocked']['color'],
            'SB': COHORT_STYLE['str']['color']}


def load_datasets(**kw):
    cohorts = load_cohorts(**kw)
    return {'UU': load_groups(['unblocked'], **kw)['unblocked'], 'UB': cohorts['unstr'], 'SB': cohorts['str']}


training = {k: df[df.week_task <= MAX_WEEK] for k, df in load_datasets(after_day=0, min_blocks=None).items()}
PERIODS = {'training': (training, f'Weeks 1-{MAX_WEEK}', False), 'expert': (load_datasets(), 'Expert data', True)}


def sessions_per_week(df):
    """Per rat: sessions, weeks with at least one session, and sessions per week."""
    s = df.drop_duplicates(['animal', 'session']).groupby('animal').agg(
        sessions=('session', 'size'), weeks=('week_task', 'nunique'))
    return s.assign(per_week=s.sessions / s.weeks).sort_values('per_week')


# %% Training, week by week: sessions each rat ran in each week (rats in the same place in every panel)
counts = {k: df.drop_duplicates(['animal', 'session']).groupby(['week_task', 'animal']).size().unstack()
          for k, df in training.items()}  # weeks x rats; NaN = no session that week
fig, axes = plt.subplots(MAX_WEEK // 2, 2, figsize=(18, 2.6 * MAX_WEEK // 2), sharex=True, sharey=True)
for week, ax in zip(range(1, MAX_WEEK + 1), axes.flat):
    x0, ticks, names = 0, [], []
    for k, c in counts.items():
        rats = sorted(c.columns)
        s = c.reindex(index=[week], columns=rats).iloc[0]  # NaN: the rat ran no session that week
        x = range(x0, x0 + len(rats))
        ax.bar(x, s.fillna(0), 0.7, color=DATASETS[k], label=k)
        for name, y, ls in [('mean', s.mean(), '-'), ('median', s.median(), '--')]:
            ax.hlines(y, x0 - 0.4, x0 + len(rats) - 0.6, color='k', ls=ls, lw=1.2)
        ax.text(x0 + (len(rats) - 1) / 2, 23, f'n = {s.notna().sum()}, mean = {s.mean():.1f}, median = {s.median():.1f}',
                ha='center', va='bottom', fontsize=7)
        ticks += list(x); names += rats
        x0 += len(rats) + 2
    ax.set(title=f'Week {week}', ylim=(0, 28), xlim=(-1, x0))
for ax in axes[-1]:
    ax.set_xticks(ticks, names, rotation=70, ha='right', fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel('Sessions')
axes[0, -1].legend(frameon=False, labelcolor='linecolor', loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
save_figure(fig, 'sessions_by_week_training', source_data=pd.concat(counts, names=['dataset']))

# %% Pooled over the period: all rats, grouped and coloured by dataset
for period, (datasets, title, show_min) in PERIODS.items():
    table = {k: sessions_per_week(df) for k, df in datasets.items()}
    fig, ax = plt.subplots(figsize=(16, 4.5))
    x0, ticks, names = 0, [], []
    for k, s in table.items():
        x = range(x0, x0 + len(s))
        ax.bar(x, s.per_week, 0.7, color=DATASETS[k], label=f'{k}')
        lines = {'mean': (s.per_week.mean(), '-'), 'median': (s.per_week.median(), '--')}
        if show_min:
            lines['min'] = (s.per_week.min(), ':')
        for name, (y, ls) in lines.items():
            ax.hlines(y, x0 - 0.4, x0 + len(s) - 0.6, color='k', ls=ls, lw=1.2)
        ax.text(x0 + (len(s) - 1) / 2, 23, '\n'.join(f'{n} = {y:.1f}' for n, (y, _) in lines.items()),
                ha='center', va='bottom', fontsize=8)
        ticks += list(x); names += list(s.index)
        x0 += len(s) + 2
    ax.set_xticks(ticks, names, rotation=70, ha='right', fontsize=8)
    ax.set(ylabel='Sessions per week', title=title, xlim=(-1, x0), ylim=(0, 28))
    ax.legend(frameon=False, labelcolor='linecolor', loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save_figure(fig, f'sessions_per_week_{period}', source_data=table)
    for k, s in table.items():
        print(period, k, len(s), s.per_week.describe()[['mean', '50%', 'min', 'max']].round(2).to_dict())

# %% Expert data, all rats: total sessions after day EXPERT_AFTER_DAY. Rats left out of the expert plots
# (MIN_BLOCKS or fewer blocks of at least N_TRIALS trials) are hatched; rats with no expert session show 0.
# Mean, median and minimum are over all rats of the dataset.
expert_all = load_datasets(min_blocks=None)
expert_rats = {k: set(df.animal) for k, df in load_datasets().items()}
fig, ax = plt.subplots(figsize=(16, 4.5))
x0, ticks, names, table = 0, [], [], {}
for k, df in expert_all.items():
    s = (df.drop_duplicates(['animal', 'session']).groupby('animal').size()
         .reindex(sorted(training[k].animal.unique()), fill_value=0).sort_values())
    table[k] = s.to_frame('sessions').assign(in_expert_plots=s.index.isin(list(expert_rats[k])))
    x = list(range(x0, x0 + len(s)))
    for xi, (rat, n) in zip(x, s.items()):
        kept = rat in expert_rats[k]
        ax.bar(xi, n, 0.7, color=DATASETS[k] if kept else 'none', edgecolor=DATASETS[k], hatch=None if kept else '///')
    ax.bar(0, 0, color=DATASETS[k], label=k)  # legend entry
    lines = {'mean': (s.mean(), '-'), 'median': (s.median(), '--'), 'min': (s.min(), ':')}
    for name, (y, ls) in lines.items():
        ax.hlines(y, x0 - 0.4, x0 + len(s) - 0.6, color='k', ls=ls, lw=1.2)
    ax.text(x0 + (len(s) - 1) / 2, 440, '\n'.join(f'{n} = {y:.0f}' for n, (y, _) in lines.items()),
            ha='center', va='bottom', fontsize=8)
    ticks += x; names += list(s.index)
    x0 += len(s) + 2
ax.bar(0, 0, color='none', edgecolor='grey', hatch='///', label='not in expert plots')
ax.set_xticks(ticks, names, rotation=70, ha='right', fontsize=8)
ax.set(ylabel='Sessions', title='Expert data: total sessions per rat', xlim=(-1, x0), ylim=(0, 520))
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
save_figure(fig, 'sessions_total_expert', source_data=table)
for k, t in table.items():
    print('expert total', k, len(t), t.sessions.describe()[['mean', '50%', 'min', 'max']].round(1).to_dict())

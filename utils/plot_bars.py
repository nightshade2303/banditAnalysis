"""Bars of per-rat values: mean across rats +- SEM, with chance lines and one dot per rat."""
import numpy as np

from utils.plotSettings import REWARD_STYLE

ERROR_KW = dict(lw=1.5, capsize=0)           # SEM error bars
CHANCE_KW = dict(color='k', linestyle='--')  # chance line on each bar


def plot_outcome_bars(ax, values, cohort_styles, chance=None, width=0.38):
    """A rewarded and an unrewarded bar for each cohort, with one dot per rat.

    values: {cohort: rats x reward table with columns 1 and 0}; chance: same shape, drawn as a dashed
    line on each bar (mean across rats)."""
    for i, (cohort, table) in enumerate(values.items()):
        for reward, offset in [(1, -width / 2), (0, width / 2)]:
            v = table[reward].dropna()
            x = i + offset
            ax.bar(x, v.mean(), width, yerr=v.sem(), color=REWARD_STYLE[reward]['color'],
                   label=REWARD_STYLE[reward]['label'] if i == 0 else None, error_kw=ERROR_KW)
            ax.scatter(np.full(len(v), x), v, s=10, color='k', alpha=0.35, zorder=3, linewidths=0)
            if chance is not None:
                ax.hlines(chance[cohort][reward].mean(), x - width / 2, x + width / 2, linewidth=1.5, zorder=4,
                          **{**CHANCE_KW, 'color': 'grey'})
    ax.set_xticks(range(len(values)), [f"{cohort_styles[c]['label']}\n(n = {len(t)})" for c, t in values.items()])
    ax.set_xlim(-0.6, len(values) - 0.4)


def _distance_stats(probs):
    """Per distance: mean and SEM across rats of p_switch, and the mean chance."""
    return probs.groupby('distance').agg(mean=('p_switch', 'mean'), sem=('p_switch', 'sem'),
                                         chance=('chance', 'mean'))


def plot_distance_bars(ax, probs, width=0.38):
    """P(switch to a port at distance d) after rewarded and unrewarded trials: mean across rats +- SEM, dashed
    line = chance. probs: one cohort's table from switch_distance_probs."""
    for reward, offset in [(1, -width / 2), (0, width / 2)]:
        stats = _distance_stats(probs[probs.reward == reward])
        x = stats.index.to_numpy() + offset
        ax.bar(x, stats['mean'], width, yerr=stats['sem'], color=REWARD_STYLE[reward]['color'],
               label=REWARD_STYLE[reward]['label'], error_kw=ERROR_KW)
        ax.hlines(stats['chance'], x - width / 2, x + width / 2, linewidth=1,
                  label='Chance' if reward == 1 else None, **CHANCE_KW)
    ax.set_xticks(sorted(probs.distance.unique()))


def plot_distance_bars_any_outcome(ax, probs, color, width=0.6, dot_color='0.3', dot_size=25, jitter=0.08,
                                   seed=0):
    """P(switch to a port at distance d) whatever the outcome: mean across rats +- SEM, one dot per rat
    (dot_color, dot_size in pt^2, spread horizontally by up to +-jitter), dashed line = chance.
    probs: one cohort's table from switch_distance_probs(by_outcome=False)."""
    stats = _distance_stats(probs)
    x = stats.index.to_numpy()
    ax.bar(x, stats['mean'], width, yerr=stats['sem'], color=color, error_kw=ERROR_KW)
    spread = np.random.default_rng(seed).uniform(-jitter, jitter, len(probs))
    ax.scatter(probs.distance + spread, probs.p_switch, s=dot_size, color=dot_color, zorder=3, linewidths=0)
    ax.hlines(stats['chance'], x - width / 2, x + width / 2, linewidth=1, label='Chance', **CHANCE_KW)
    ax.set_xticks(x)

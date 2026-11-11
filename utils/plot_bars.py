"""Bars of per-rat values: mean across rats +- SEM, rewarded (red) next to unrewarded (blue)."""
import numpy as np

from utils.plotSettings import REWARD_STYLE


def plot_outcome_bars(ax, values, cohort_styles, chance=None, width=0.38):
    """A rewarded and an unrewarded bar for each cohort, with one dot per rat (notebook cells 113, 127).

    values: {cohort: rats x reward table with columns 1 and 0}; chance: same shape, drawn as a dashed
    line on each bar (mean across rats)."""
    for i, (cohort, table) in enumerate(values.items()):
        for reward, offset in [(1, -width / 2), (0, width / 2)]:
            v = table[reward].dropna()
            x = i + offset
            ax.bar(x, v.mean(), width, yerr=v.sem(), color=REWARD_STYLE[reward]['color'],
                   label=REWARD_STYLE[reward]['label'] if i == 0 else None, error_kw=dict(lw=1.5, capsize=0))
            ax.scatter(np.full(len(v), x), v, s=10, color='k', alpha=0.35, zorder=3, linewidths=0)
            if chance is not None:
                ax.hlines(chance[cohort][reward].mean(), x - width / 2, x + width / 2,
                          color='grey', linestyle='--', linewidth=1.5, zorder=4)
    ax.set_xticks(range(len(values)), [f"{cohort_styles[c]['label']}\n(n = {len(t)})" for c, t in values.items()])
    ax.set_xlim(-0.6, len(values) - 0.4)


def plot_distance_bars(ax, probs, width=0.38):
    """P(switch to a port at distance d) after rewarded and unrewarded trials (notebook cell 137): mean
    across rats +- SEM, dashed line = chance. probs: one cohort's table from switch_distance_probs."""
    for reward, offset in [(1, -width / 2), (0, width / 2)]:
        stats = probs[probs.reward == reward].groupby('distance').agg(
            mean=('p_switch', 'mean'), sem=('p_switch', 'sem'), chance=('chance', 'mean'))
        x = stats.index.to_numpy() + offset
        ax.bar(x, stats['mean'], width, yerr=stats['sem'], color=REWARD_STYLE[reward]['color'],
               label=REWARD_STYLE[reward]['label'], error_kw=dict(lw=1.5, capsize=0))
        ax.hlines(stats['chance'], x - width / 2, x + width / 2, color='k', linestyle='--', linewidth=1,
                  label='Chance' if reward == 1 else None)
    ax.set_xticks(stats.index)


def plot_distance_bars_any_outcome(ax, probs, color, width=0.6):
    """P(switch to a port at distance d) whatever the outcome: mean across rats +- SEM, one dot per rat,
    dashed line = chance. probs: one cohort's table from switch_distance_probs(by_outcome=False)."""
    stats = probs.groupby('distance').agg(mean=('p_switch', 'mean'), sem=('p_switch', 'sem'), chance=('chance', 'mean'))
    x = stats.index.to_numpy()
    ax.bar(x, stats['mean'], width, yerr=stats['sem'], color=color, error_kw=dict(lw=1.5, capsize=0))
    ax.scatter(probs.distance, probs.p_switch, s=10, color='k', alpha=0.35, zorder=3, linewidths=0)
    ax.hlines(stats['chance'], x - width / 2, x + width / 2, color='k', linestyle='--', linewidth=1, label='Chance')
    ax.set_xticks(x)

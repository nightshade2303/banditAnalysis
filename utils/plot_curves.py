"""Mean +- SEM across rats of trial-by-trial curves (rats x trials tables from utils/trial_curves.py)."""
import matplotlib.pyplot as plt
import numpy as np

from utils.trial_curves import segment_edges


def plot_mean_sem(ax, curves, color, label):
    """Mean across rats with an SEM band; x = trial."""
    mean, err = curves.mean(), curves.sem()
    ax.plot(curves.columns, mean, color=color, label=label)
    ax.fill_between(curves.columns, mean - err, mean + err, color=color, alpha=0.2, lw=0)


def plot_block_segments(ax, curves, styles, segments, gap=2.5):
    """Mean +- SEM of block segments laid side by side (notebook cells 44, 103, 135).

    curves: {cohort: rats x trials}, styles: {cohort: dict(color=..., label=...)}. The x-axis is broken
    where the middle of a block is skipped and a dashed line marks the block switch. Ticks give the trial
    within the block; negative ticks count back from the block's end."""
    edges = segment_edges(segments)
    n = edges[-1]
    inner = list(zip(segments[1:], edges[1:-1]))
    breaks = [e for (_, which, _), e in inner if which == 'last']
    switches = [e for (_, which, _), e in inner if which == 'first']

    def xmap(t):  # shift x after each break so the curves leave a gap there
        return np.asarray(t) + 2 * gap * np.searchsorted(breaks, t, side='right')

    x = xmap(np.arange(n))
    runs = list(zip([0] + breaks, breaks + [n]))
    for cohort, c in curves.items():
        mean, err = c.mean().to_numpy(), c.sem().to_numpy()
        color, label = styles[cohort]['color'], styles[cohort]['label']
        for i, (s, e) in enumerate(runs):
            ax.plot(x[s:e], mean[s:e], color=color, label=label if i == 0 else None)
            ax.fill_between(x[s:e], mean[s:e] - err[s:e], mean[s:e] + err[s:e], color=color, alpha=0.2, lw=0)

    for b in switches:
        ax.axvline(xmap(b), color='grey', linestyle='--', linewidth=0.75)
    _draw_axis_breaks(ax, xmap(breaks) - gap - 0.5, gap)

    ticks = list(edges[:-1]) + [n - 1]
    labels = ['1' if which == 'first' else f'−{k}' for _, which, k in segments]
    labels.append(str(segments[-1][2]) if segments[-1][1] == 'first' else '−1')
    ax.set_xticks(xmap(ticks), labels)
    ax.set_xlim(-1, xmap(n - 1) + 1)
    for block in dict.fromkeys(b for b, _, _ in segments):  # block names above the axes
        cols = [i for i, (b, _, _) in enumerate(segments) if b == block]
        middle = (xmap(edges[cols[0]]) + xmap(edges[cols[-1] + 1] - 1)) / 2
        ax.text(middle, 1.01, f'Block {block}', transform=ax.get_xaxis_transform(), ha='center', va='bottom')


def _draw_axis_breaks(ax, centres, gap):
    """Cut the x-axis spine at each centre and mark the cut with two slashes (notebook cell 44)."""
    trans = ax.get_xaxis_transform()  # x in data units, y in axes units
    for b in centres:
        ax.add_patch(plt.Rectangle((b - gap, -0.02), 2 * gap, 0.04, transform=trans,
                                   facecolor=ax.figure.get_facecolor(), edgecolor='none', clip_on=False, zorder=3))
        ax.plot([b - gap, b + gap], [0, 0], transform=trans, linestyle='none',
                marker=[(-1, -1.5), (1, 1.5)], markersize=8, markeredgewidth=1,
                color='k', clip_on=False, zorder=4)

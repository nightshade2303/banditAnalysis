"""Mean +- SEM across rats of trial-by-trial curves (rats x trials tables from utils/trial_curves.py), the
standard figures built from them, chance lines, and per-rat bars for block 1 vs block 2."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from utils.config import N_TRIALS, TRIAL_TICKS, TRIAL_XLABEL
from utils.trial_curves import segment_edges

SEM_ALPHA = 0.2           # opacity of the SEM band
CHANCE_STYLE = dict(linestyle='--', linewidth=0.75)
EVERY_BLOCK_FIGSIZE = (5, 4.5)
SEGMENTS_FIGSIZE = (9, 4.5)


def plot_mean_sem(ax, curves, color, label):
    """Mean across rats with an SEM band; x = trial."""
    mean, err = curves.mean(), curves.sem()
    ax.plot(curves.columns, mean, color=color, label=label)
    ax.fill_between(curves.columns, mean - err, mean + err, color=color, alpha=SEM_ALPHA, lw=0)


def format_trial_axis(ax):
    """x axis of a plot over the first N_TRIALS trials of a block."""
    ax.set(xlabel=TRIAL_XLABEL, xlim=(1, N_TRIALS), xticks=TRIAL_TICKS)


def plot_chance(ax, levels, label=False):
    """Dashed chance line(s). levels: [(chance, colour of its curve), ...]. If all levels are equal, one black
    line; otherwise one line per level in its curve's colour. label=True writes 'Chance' just above the
    (single) line, at the right."""
    values = [level for level, _ in levels]
    if np.allclose(values, values[0]):
        ax.axhline(values[0], color='k', **CHANCE_STYLE)
        if label:
            ax.text(0.98, values[0], 'Chance', transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                    fontsize='small')
    else:
        for level, color in levels:
            ax.axhline(level, color=color, **CHANCE_STYLE)


def ordered_legend(ax, first=None, **kwargs):
    """Legend with labels coloured as their lines, the labels in `first` (in that order) listed first."""
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (first or []).index(labels[i])
                   if labels[i] in (first or []) else len(labels) + i)
    return ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
                     labelcolor='linecolor', **kwargs)


def every_block_figure(curves, styles, ylabel, legend_first=None, legend_loc='best'):
    """Figure of mean +- SEM curves over the first N_TRIALS trials of a block. curves: {key: rats x trials};
    styles: {key: dict(color=..., label=...)}. Returns fig, ax."""
    fig, ax = plt.subplots(figsize=EVERY_BLOCK_FIGSIZE)
    for key, rats_x_trials in curves.items():
        plot_mean_sem(ax, rats_x_trials, **styles[key])
    format_trial_axis(ax)
    ax.set_ylabel(ylabel)
    ordered_legend(ax, legend_first, loc=legend_loc)
    return fig, ax


def block_segments_figure(curves, styles, segments, ylabel, legend_first=None, legend_loc='upper right'):
    """Figure of block segments side by side (block 1 -> block 2), see plot_block_segments. Returns fig, ax."""
    fig, ax = plt.subplots(figsize=SEGMENTS_FIGSIZE)
    plot_block_segments(ax, curves, styles, segments)
    ax.set(xlabel=TRIAL_XLABEL, ylabel=ylabel)
    ordered_legend(ax, legend_first, loc=legend_loc)
    return fig, ax


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
            ax.fill_between(x[s:e], mean[s:e] - err[s:e], mean[s:e] + err[s:e], color=color, alpha=SEM_ALPHA,
                            lw=0)

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


def block_means(curves, segments, blocks=(1, 2)):
    """Per rat, the mean over the trials of the 'first' segment of each block (unsmoothed rats x trials
    from segment_matrix), e.g. trials 1-100 of block 1 and of block 2 for BLOCK_1_TO_2."""
    edges = segment_edges(segments)
    cols = {b: (edges[i], edges[i + 1]) for i, (b, which, _) in enumerate(segments) if which == 'first'}
    return pd.DataFrame({f'Block {b}': curves.iloc[:, cols[b][0]:cols[b][1]].mean(axis=1) for b in blocks})


def plot_block_bars(ax, means, styles):
    """Bars (mean +- SEM across rats) grouped by block, one bar per cohort, with each rat's value as a dot.
    means: {cohort: rats x blocks}, two cohorts. In each block the cohorts are compared with a Mann-Whitney U
    test, whose p is written above the block. Returns the tests as a table."""
    (c1, a), (c2, b) = list(means.items())[:2]
    n_bars = len(means)
    rng = np.random.default_rng(0)  # horizontal jitter of the dots
    tests, centres = [], []
    for j, block in enumerate(a.columns):
        first_x = j * (n_bars + 1)  # one empty slot between blocks
        for i, (cohort, m) in enumerate(means.items()):
            v = m[block].dropna()
            color = styles[cohort]['color']
            ax.bar(first_x + i, v.mean(), 0.7, yerr=v.sem(), color=color, alpha=0.5,
                   error_kw=dict(lw=1.5, capsize=0), label=styles[cohort]['label'] if j == 0 else None)
            ax.scatter(first_x + i + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color=color, zorder=3,
                       linewidths=0)
        u = mannwhitneyu(a[block].dropna(), b[block].dropna())
        tests.append(dict(block=block, comparison=f'{c1} vs {c2}',
                          n=f'{a[block].notna().sum()}, {b[block].notna().sum()}', U=u.statistic, p=u.pvalue))
        centres.append(first_x + (n_bars - 1) / 2)
        ax.text(centres[-1], 1.0, p_text(u.pvalue), transform=ax.get_xaxis_transform(), ha='center',
                va='bottom', fontsize=9)
    ax.set_xticks(centres, list(a.columns))
    return pd.DataFrame(tests)


def p_text(p):
    """p value as text for a figure: 'p = 0.012', or 'p < 0.001'."""
    return f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'


def _draw_axis_breaks(ax, centres, gap):
    """Cut the x-axis spine at each centre and mark the cut with two slashes (notebook cell 44)."""
    trans = ax.get_xaxis_transform()  # x in data units, y in axes units
    for b in centres:
        ax.add_patch(plt.Rectangle((b - gap, -0.02), 2 * gap, 0.04, transform=trans,
                                   facecolor=ax.figure.get_facecolor(), edgecolor='none', clip_on=False, zorder=3))
        ax.plot([b - gap, b + gap], [0, 0], transform=trans, linestyle='none',
                marker=[(-1, -1.5), (1, 1.5)], markersize=8, markeredgewidth=1,
                color='k', clip_on=False, zorder=4)

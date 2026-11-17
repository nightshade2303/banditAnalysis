"""Two-column alluvial plot (as in DABEST paired proportion plots): stacked bars on the left and right, joined by
bands whose width is the share of each right bar that comes from each left bar."""
import numpy as np
from matplotlib.patches import Rectangle

BAR_W = 0.12     # bar width, x units (left bar at x = 0, right bar at x = 1)
EDGE_W = 0.03    # width of the strip to the right of each right bar, in the right bar's colour
EDGE_GAP = 0.03  # gap between a right bar and its strip
GAP = 0.12       # vertical gap between stacked bars, y units (each right bar is 1 high)
FLOW_ALPHA = 0.35
TICK_LEN = 0.03  # length of the scale ticks on the right bars, x units


def _band(ax, x0, x1, y0_lo, y0_hi, y1_lo, y1_hi, color):
    """Band from [y0_lo, y0_hi] at x0 to [y1_lo, y1_hi] at x1, with a smooth (cosine) S-shape."""
    x = np.linspace(x0, x1, 100)
    s = (1 - np.cos(np.pi * (x - x0) / (x1 - x0))) / 2
    ax.fill_between(x, y0_lo + s * (y1_lo - y0_lo), y0_hi + s * (y1_hi - y0_hi), color=color, alpha=FLOW_ALPHA,
                    lw=0, zorder=1)


def plot_alluvial(ax, shares, left_colors, right_colors, left_labels, right_labels, pct_fmt=None,
                  right_ticks=None, vertical_labels=False, left_pad=0):
    """shares[i][j]: fraction of right bar i (top = 0) that comes from left bar j (top = 0); each row sums to 1.
    Right bars are all 1 high; a left bar is as high as everything it sends.
    left_colors[j]: left bar j, its label, its bands, its part of each right bar (and the band's percentage).
    right_colors[i]: right bar i's label and a strip on its right edge (the mixed parts make up the whole bar).
    pct_fmt: if given (e.g. '{:.0f}%'), each band is labelled with its share of the right bar, next to the bar.
    right_ticks: if given (e.g. [0, 50, 100]), a % scale on the left edge of each right bar, ticks pointing in.
    left_pad: extra x room left of the left bar (e.g. for insets), with the left labels moved out by as much.
    Returns {'left_top', 'left_h', 'right_top'}: y of each bar's top and the left bars' heights (y increases
    downwards), to place other things next to the bars."""
    shares = np.asarray(shares, float)
    n_right, n_left = shares.shape
    left_h = shares.sum(axis=0)

    def tops(heights):  # top y of each stacked bar, first bar on top
        return np.cumsum(np.r_[0, heights[:-1]]) + GAP * np.arange(len(heights))

    total = max(left_h.sum() + GAP * (n_left - 1), n_right + GAP * (n_right - 1))
    left_top, right_top = tops(left_h), tops(np.ones(n_right))
    # centre both columns vertically
    left_top += (total - left_h.sum() - GAP * (n_left - 1)) / 2
    right_top += (total - n_right - GAP * (n_right - 1)) / 2

    left_used = np.zeros(n_left)   # how much of each left bar already sent, from its top
    for i in range(n_right):       # right bars top to bottom, each filled from its top in left-bar order
        y1 = right_top[i]
        for j in range(n_left):
            h = shares[i, j]
            if h <= 0:
                continue
            y0 = left_top[j] + left_used[j]
            _band(ax, BAR_W / 2, 1 - BAR_W / 2, y0, y0 + h, y1, y1 + h, left_colors[j])
            ax.add_patch(Rectangle((1 - BAR_W / 2, y1), BAR_W, h, fc=left_colors[j], ec='none', zorder=2))
            if pct_fmt:
                ax.text(1 - BAR_W / 2 - 0.02, y1 + h / 2, pct_fmt.format(100 * h), va='center', ha='right',
                        fontsize='small', color=left_colors[j], zorder=3)
            left_used[j] += h
            y1 += h
        ax.add_patch(Rectangle((1 + BAR_W / 2 + EDGE_GAP, right_top[i]), EDGE_W, 1, fc=right_colors[i], ec='none', zorder=2))
        for t in right_ticks or []:  # 0 at the top of the bar, 100 at the bottom
            y = right_top[i] + t / 100
            ax.plot([1 - BAR_W / 2, 1 - BAR_W / 2 + TICK_LEN], [y, y], color='k', lw=0.8, zorder=3,
                    solid_capstyle='butt')
            ax.text(1 - BAR_W / 2 - 0.02, y, f'{t:g}', va='center', ha='right', fontsize='x-small', zorder=3)
    for j in range(n_left):
        ax.add_patch(Rectangle((-BAR_W / 2, left_top[j]), BAR_W, left_h[j], fc=left_colors[j], ec='none', zorder=2))

    ax.set_xlim(-0.1 - left_pad, 1.15) if vertical_labels else ax.set_xlim(-0.35 - left_pad, 1.35)
    ax.set_ylim(total, 0)  # first bar on top
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    # vertical labels: read bottom to top, along the bar, so a narrow figure keeps room for the plot
    rot = dict(rotation=90, rotation_mode='anchor', ha='center') if vertical_labels else {}
    for j, label in enumerate(left_labels):
        ax.text(-BAR_W / 2 - 0.03 - left_pad, left_top[j] + left_h[j] / 2, label, color=left_colors[j],
                **(rot | dict(va='bottom') if vertical_labels else dict(ha='right', va='center')))
    for i, label in enumerate(right_labels):
        ax.text(1 + BAR_W / 2 + EDGE_GAP + EDGE_W + 0.03, right_top[i] + 0.5, label, color=right_colors[i],
                **(rot | dict(va='top') if vertical_labels else dict(ha='left', va='center')))
    return dict(left_top=left_top, left_h=left_h, right_top=right_top)


def inset_bar_column(ax, x0, x1, top, height, rows, color, ylim, gap_before_last=True):
    """Small bar plots stacked from y = top to top + height (data units of ax, y increasing downwards), between
    x0 and x1: one per row of `rows` (bar heights). With gap_before_last, a slot of three dots before the last
    plot stands for the rows not shown. Returns the inset axes."""
    n_slots = len(rows) + gap_before_last
    slot = height / n_slots
    to_axes = ax.transData + ax.transAxes.inverted()
    insets, k = [], 0
    for i, r in enumerate(rows):
        if gap_before_last and i == len(rows) - 1:  # dots slot
            yc = top + (k + 0.5) * slot
            ax.plot([(x0 + x1) / 2] * 3, yc + np.array([-0.2, 0, 0.2]) * slot, 'o', ms=1.5, color='0.3',
                    clip_on=False)
            k += 1
        (ax0, ay0), (ax1, ay1) = to_axes.transform([(x0, top + (k + 0.9) * slot), (x1, top + (k + 0.1) * slot)])
        iax = ax.inset_axes([ax0, ay0, ax1 - ax0, ay1 - ay0])
        iax.bar(range(1, len(r) + 1), r, width=1, color=color)
        iax.set(ylim=ylim, xlim=(0.5, len(r) + 0.5), xticks=[], yticks=[])
        for side in ['top', 'right', 'left']:
            iax.spines[side].set_visible(False)
        insets.append(iax)
        k += 1
    return insets

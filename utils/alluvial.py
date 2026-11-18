"""Two-column alluvial plot (as in DABEST paired proportion plots): stacked bars on the left and right, joined by
bands whose width is the share of each right bar that comes from each left bar."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

BAR_W = 0.12     # bar width, x units (left bar at x = 0, right bar at x = 1)
EDGE_W = 0.03    # width of the strip to the right of each right bar, in the right bar's colour
EDGE_GAP = 0.03  # gap between a right bar and its strip
GAP = 0.12       # vertical gap between stacked bars, y units (each right bar is 1 high)
FLOW_ALPHA = 0.35
TICK_LEN = 0.03  # length of the scale ticks on the right bars, x units
TICK_LABEL_W = 0.12  # room for a tick number ('0.5') right of the right bars, x units
SCALE_LABEL_W = 0.09  # room for the scale label (e.g. 'Proportion') right of the tick numbers, x units


def _band(ax, x0, x1, y0_lo, y0_hi, y1_lo, y1_hi, color):
    """Band from [y0_lo, y0_hi] at x0 to [y1_lo, y1_hi] at x1, with a smooth (cosine) S-shape."""
    x = np.linspace(x0, x1, 100)
    s = (1 - np.cos(np.pi * (x - x0) / (x1 - x0))) / 2
    ax.fill_between(x, y0_lo + s * (y1_lo - y0_lo), y0_hi + s * (y1_hi - y0_hi), color=color, alpha=FLOW_ALPHA,
                    lw=0, zorder=1)


def plot_alluvial(ax, shares, left_colors, right_colors, left_labels, right_labels, pct_fmt=None,
                  right_ticks=None, vertical_labels=False, left_pad=0, width=1, left_bars=True, right_strip=True,
                  gap=GAP, ticks_side='left', scale_label=None, right_label_rotation=90,
                  scale_label_pad=TICK_LABEL_W, scale_label_size='small', ticks_up=False,
                  right_minor_ticks=None):
    """shares[i][j]: fraction of right bar i (top = 0) that comes from left bar j (top = 0); each row sums to 1.
    Right bars are all 1 high; a left bar is as high as everything it sends.
    left_colors[j]: left bar j, its label, its bands, its part of each right bar (and the band's percentage).
    right_colors[i]: right bar i's label and a strip on its right edge (the mixed parts make up the whole bar).
    pct_fmt: if given (e.g. '{:.0f}%'), each band is labelled with its share of the right bar, next to the bar.
    right_ticks: if given (e.g. [0, 0.5, 1]), a scale (fraction of the bar) on the left edge of each right bar,
    each tick crossing the edge (half over the bands, half over the bar), numbers to its left. 0 at the top of
    each bar, or at the bottom with ticks_up. right_minor_ticks: more ticks on that scale, without numbers.
    scale_label: with ticks_side='right', a label for the scale (e.g. 'Proportion'), right of the tick numbers,
    centred on each right bar and turned as the right labels; scale_label_pad: room left for the tick numbers
    (x units), scale_label_size: its font size.
    right_label_rotation: 90 (read bottom to top) or 270 (top to bottom), for vertical_labels.
    left_pad: extra x room left of the left bar (e.g. for insets), with the left labels moved out by as much.
    width: distance between the left and right bar centres, x units (bars, strip and gaps keep their size).
    left_bars=False: no left bars; the bands start where the left bar's left edge would be.
    right_strip=False: no environment-coloured strip right of the right bars (the right labels stay).
    gap: vertical gap between stacked bars in both columns, y units (each right bar is 1 high).
    ticks_side: right_ticks on the 'left' edge of the right bars (crossing it, numbers left) or on the 'right'
    edge (pointing out, numbers right; the right labels move out to make room).
    Also returns 'left_texts', the left label Text artists.
    Returns {'left_top', 'left_h', 'right_top'}: y of each bar's top and the left bars' heights (y increases
    downwards), to place other things next to the bars."""
    shares = np.asarray(shares, float)
    n_right, n_left = shares.shape
    left_h = shares.sum(axis=0)

    def tops(heights):  # top y of each stacked bar, first bar on top
        return np.cumsum(np.r_[0, heights[:-1]]) + gap * np.arange(len(heights))

    total = max(left_h.sum() + gap * (n_left - 1), n_right + gap * (n_right - 1))
    left_top, right_top = tops(left_h), tops(np.ones(n_right))
    # centre both columns vertically
    left_top += (total - left_h.sum() - gap * (n_left - 1)) / 2
    right_top += (total - n_right - gap * (n_right - 1)) / 2

    left_used = np.zeros(n_left)   # how much of each left bar already sent, from its top
    for i in range(n_right):       # right bars top to bottom, each filled from its top in left-bar order
        y1 = right_top[i]
        for j in range(n_left):
            h = shares[i, j]
            if h <= 0:
                continue
            y0 = left_top[j] + left_used[j]
            _band(ax, BAR_W / 2 if left_bars else -BAR_W / 2, width - BAR_W / 2, y0, y0 + h, y1, y1 + h, left_colors[j])
            ax.add_patch(Rectangle((width - BAR_W / 2, y1), BAR_W, h, fc=left_colors[j], ec='none', zorder=2))
            if pct_fmt:
                ax.text(width - BAR_W / 2 - 0.02, y1 + h / 2, pct_fmt.format(100 * h), va='center', ha='right',
                        fontsize='small', color=left_colors[j], zorder=3)
            left_used[j] += h
            y1 += h
        if right_strip:
            ax.add_patch(Rectangle((width + BAR_W / 2 + EDGE_GAP, right_top[i]), EDGE_W, 1, fc=right_colors[i],
                                   ec='none', zorder=2))
        labelled = set(right_ticks or [])
        for t in sorted(labelled | set(right_minor_ticks or [])):  # 0 at the top of the bar and 1 at the bottom,
            y = right_top[i] + (1 - t if ticks_up else t)             # or the other way round
            number = f'{t:g}' if t in labelled else ''
            if ticks_side == 'left':  # tick crosses the band/bar edge, number to its left
                edge = width - BAR_W / 2
                ax.plot([edge - TICK_LEN, edge + TICK_LEN], [y, y], color='k', lw=0.8, zorder=3,
                        solid_capstyle='butt')
                ax.text(edge - TICK_LEN - 0.02, y, number, va='center', ha='right', fontsize='xx-small',
                        color='grey', zorder=3)
            else:  # outward tick on the bar's right edge, number to its right
                edge = width + BAR_W / 2
                ax.plot([edge, edge + TICK_LEN], [y, y], color='k', lw=0.8, zorder=3, solid_capstyle='butt')
                ax.text(edge + TICK_LEN + 0.02, y, number, va='center', ha='left', fontsize='xx-small',
                        color='grey', zorder=3)
    for j in range(n_left if left_bars else 0):
        ax.add_patch(Rectangle((-BAR_W / 2, left_top[j]), BAR_W, left_h[j], fc=left_colors[j], ec='none', zorder=2))

    # room for the tick numbers between a right bar and its label when the ticks are on the right
    ticks_right = right_ticks and ticks_side == 'right'
    right_pad = TICK_LEN + 0.02 + TICK_LABEL_W if ticks_right else 0
    # vertical labels run along the bar, so a narrow figure keeps room for the plot. With rotation_mode
    # 'anchor', va='top' at rotation 90 and va='bottom' at 270 both put the text right of its x.
    right_rot = dict(rotation=right_label_rotation, rotation_mode='anchor', ha='center',
                     va='top' if right_label_rotation % 360 == 90 else 'bottom')
    if ticks_right and scale_label:
        right_pad += scale_label_pad - TICK_LABEL_W
        for top in right_top:  # one label per bar, centred on it
            ax.text(width + BAR_W / 2 + right_pad, top + 0.5, scale_label, fontsize=scale_label_size, **right_rot)
        right_pad += SCALE_LABEL_W
    ax.set_xlim(-0.1 - left_pad, width + 0.15 + right_pad) if vertical_labels else \
        ax.set_xlim(-0.35 - left_pad, width + 0.35 + right_pad)
    ax.set_ylim(total, 0)  # first bar on top
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    left_rot = dict(rotation=90, rotation_mode='anchor', ha='center', va='bottom')
    left_texts = [ax.text(-BAR_W / 2 - 0.03 - left_pad, left_top[j] + left_h[j] / 2, label, color=left_colors[j],
                          **(left_rot if vertical_labels else dict(ha='right', va='center')))
                  for j, label in enumerate(left_labels)]
    for i, label in enumerate(right_labels):
        ax.text(width + BAR_W / 2 + (EDGE_GAP + EDGE_W if right_strip else 0) + right_pad + 0.03,
                right_top[i] + 0.5, label, color=right_colors[i],
                **(right_rot if vertical_labels else dict(ha='left', va='center')))
    return dict(left_top=left_top, left_h=left_h, right_top=right_top, left_texts=left_texts)


def rounded_box(ax, x0, x1, y0, y1, color, radius=0.04, lw=1):
    """Unfilled box with rounded corners from x0 to x1 and y0 to y1 (data units of ax). radius is in x units;
    the corners are round on the page whatever the axes' aspect (call after the layout is final)."""
    bbox = ax.get_window_extent()
    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    aspect = (abs(yb - ya) / bbox.height) / (abs(xb - xa) / bbox.width)  # y units per x unit, on the page
    y0, y1 = min(y0, y1), max(y0, y1)
    return ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f'round,pad=0,rounding_size={radius}',
                                       mutation_aspect=aspect, fc='none', ec=color, lw=lw, zorder=3, clip_on=False))


def inset_bar_column(ax, x0, x1, top, height, rows, color, ylim, gap_before_last=True, xticklabels_last=True,
                     fontsize='x-small', fill=0.7, axis_lw=0.5, peak_label='{:g}%', peak_color='0.25',
                     peak_fontsize='xx-small', alpha=1, dots_frac=0.5, yticks=None, yticklabels_first=False):
    """Small bar plots stacked from y = top to top + height (data units of ax, y increasing downwards), between
    x0 and x1: one per row of `rows` (bar heights, x = 1, 2, ...). color: one colour, or {bar value: colour}.
    With gap_before_last, a slot of three dots (dots_frac of a plot's slot high) before the last plot stands for
    the rows not shown. x ticks 1..n, labelled only on the last plot if xticklabels_last. A y axis with these
    yticks (numbers in peak_fontsize; on the first plot only with yticklabels_first) if given, else none.
    peak_label (None: off) writes the highest bar's value
    inside its top, in peak_color. Each plot takes `fill` of its slot's height. Returns the inset axes."""
    dots = dots_frac if gap_before_last else 0
    slot = height / (len(rows) + dots)
    to_axes = ax.transData + ax.transAxes.inverted()
    insets, k = [], 0.0  # k: slots used so far, from the top
    for i, r in enumerate(rows):
        if dots and i == len(rows) - 1:  # dots slot
            yc = top + (k + dots / 2) * slot
            ax.plot([(x0 + x1) / 2] * 3, yc + np.array([-0.3, 0, 0.3]) * dots * slot, 'o', ms=1.5, color='0.3',
                    clip_on=False)
            k += dots
        y_hi, y_lo = top + (k + (1 + fill) / 2) * slot, top + (k + (1 - fill) / 2) * slot
        (ax0, ay0), (ax1, ay1) = to_axes.transform([(x0, y_hi), (x1, y_lo)])
        iax = ax.inset_axes([ax0, ay0, ax1 - ax0, ay1 - ay0])
        x = range(1, len(r) + 1)
        iax.bar(x, r, width=1, alpha=alpha, color=[color[v] for v in r] if isinstance(color, dict) else color)
        iax.set(ylim=ylim, xlim=(0.5, len(r) + 0.5), xticks=list(x), yticks=yticks or [])
        iax.tick_params(labelsize=fontsize, length=2, width=axis_lw, pad=1,
                        labelbottom=xticklabels_last and i == len(rows) - 1)
        iax.tick_params(axis='y', labelsize=peak_fontsize, labelleft=not yticklabels_first or i == 0)
        if yticks:  # numbers as '0', '0.8', centred on their ticks
            iax.set_yticklabels([f'{t:g}' for t in yticks])
        for side in ['top', 'right'] + ([] if yticks else ['left']):
            iax.spines[side].set_visible(False)
        for side in ['bottom', 'left']:
            iax.spines[side].set_linewidth(axis_lw)
        if peak_label:
            p = int(np.argmax(r))
            iax.text(p + 1, r[p], peak_label.format(r[p]), ha='center', va='top', fontsize=peak_fontsize,
                     color=peak_color)
        insets.append(iax)
        k += 1
    return insets

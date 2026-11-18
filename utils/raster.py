"""Choice raster of one rat's trials (rodentBanditAnalysis.ipynb, choice raster cell): a tick per trial at the
chosen port (rewarded: tall, thick and black; unrewarded: short, thin and grey) over background bands whose
shade gives each port's reward probability in that block."""
import numpy as np
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from utils.config import PORTS
from utils.plotSettings import BLOCK_STYLE
from utils.task_profiles import REWARD_PERCENTS, parse_profile

SHADE_RANGE = (0.25, 1.0)  # part of the white -> colour map used, lowest -> highest reward %
RECT_ALPHA = 0.7
REW_COLOR, UNREW_COLOR = 'k', '0.4'
REW_LW, UNREW_LW = 0.8, 0.8   # about one trial wide at ~600 trials in 7 inches, so ticks don't merge
REW_H, UNREW_H = 0.3, 0.12    # half-heights: rewarded 3/4 of the band (0.8), unrewarded short
SCALEBAR_FRAC = 0.1           # scale bar: largest 1/2/5 x 10^k trials within this fraction of the x-range


def shades(color):
    """{reward %: shade} from white to color: each reward % of the task (REWARD_PERCENTS) placed along
    SHADE_RANGE in proportion to its value, so a reward % gets the same shade in every plot."""
    lo, hi = SHADE_RANGE
    cmap = LinearSegmentedColormap.from_list('band', ['white', color])
    percents = np.array(REWARD_PERCENTS)
    frac = (percents - percents.min()) / (percents.max() - percents.min())
    return dict(zip(REWARD_PERCENTS, cmap(lo + frac * (hi - lo))))


def nice_length(span, frac=SCALEBAR_FRAC):
    """Largest 1, 2 or 5 x 10^k that is <= frac * span."""
    target = max(span * frac, 1)
    mag = 10 ** np.floor(np.log10(target))
    for m in (5, 2, 1):
        if m * mag <= target:
            return m * mag
    return mag


def plot_raster(ax, trials, color, color_by_env=False, env_style=None):
    """Raster of `trials` (rows of one rat from utils/load_data.py) on ax, x = trial.

    color: band colour for the rat's dataset. color_by_env=True colours each block by its own reward profile
    instead (env column), with env_style (default BLOCK_STYLE): correlated profile env_style['str'], any other
    env_style['unstr'].
    The legend sits above the axes; returns it so the caller can leave room for it (see fit_legend)."""
    env_style = env_style or BLOCK_STYLE
    rew, unrew = trials[trials.reward == 1], trials[trials.reward == 0]  # no outcome (NaN): no tick
    ax.vlines(unrew.trial, unrew.port - UNREW_H, unrew.port + UNREW_H, color=UNREW_COLOR, lw=UNREW_LW, zorder=3)
    ax.vlines(rew.trial, rew.port - REW_H, rew.port + REW_H, color=REW_COLOR, lw=REW_LW, zorder=4)

    env_colors = {env: env_style[env]['color'] for env in ['str', 'unstr']}  # block-type colours
    palettes, used = {}, []
    blocks = trials.groupby('block_num').agg(start=('trial', 'min'), end=('trial', 'max'),
                                             probs=('rewprobfull', 'first'), env=('env', 'first'))
    for _, b in blocks.iterrows():
        c = env_colors[b.env] if color_by_env else color
        if c not in palettes:
            palettes[c] = shades(c)
            used.append(b.env if color_by_env else None)
        for port, p in zip(PORTS, parse_profile(b.probs)):
            if p in palettes[c]:  # bands a little shorter than 1, so port rows are separated by white
                ax.add_patch(Rectangle((b.start - 0.5, port - 0.4), b.end - b.start + 1, 0.8,
                                       fc=palettes[c][p], alpha=RECT_ALPHA, ec='none', zorder=0))
        ax.axvline(b.end + 0.5, color='white', lw=1.5, zorder=1)

    ax.set_ylim(0.5, 4.5)
    ax.invert_yaxis()  # port 1 on top
    ax.set_yticks(PORTS, [f'Port {p}' for p in PORTS])
    ax.tick_params(left=False, labelsize='small')
    ax.set_xticks([])
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)
    _add_scalebar(ax)
    ax.set_xlim(trials.trial.min() - 0.5, trials.trial.max() + 0.5)  # also draws the scale bar

    # legend above the plot as a grid: one row of shades per colour used, then the tick types
    def blank(label=''):
        return Patch(fc='none', ec='none', label=label)

    # rows: [colour label,] 10% 15% 46% 80%, then a last column with Rewarded (row 1) and Unrewarded (row 2)
    outcome = [Line2D([], [], color=REW_COLOR, marker='|', ls='none', ms=8, mew=1.5, label='Rewarded'),
               Line2D([], [], color=UNREW_COLOR, marker='|', ls='none', ms=3.5, mew=1.5, label='Unrewarded')]
    rows = []
    for c, env in zip(palettes, used):
        shade_handles = [Patch(fc=s, alpha=RECT_ALPHA, ec='none', label=f'{p:g}%') for p, s in palettes[c].items()]
        rows.append(([blank(env_style[env]['label'])] if color_by_env else []) + shade_handles)
    while len(rows) < len(outcome):
        rows.append([blank()] * len(rows[0]))
    rows = [r + [outcome[i] if i < len(outcome) else blank()] for i, r in enumerate(rows)]
    ncol = len(rows[0])
    handles = [rows[i][j] for j in range(ncol) for i in range(len(rows))]  # legend fills column by column
    return ax.legend(handles=handles, ncol=ncol, loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False,
                     handlelength=1.0, handletextpad=0.4, columnspacing=0.9, labelspacing=0.3, fontsize='small')


def fit_legend(fig, legend, base_height):
    """Make fig base_height inches plus the legend's height; tight_layout keeps the legend inside the figure."""
    fig.canvas.draw()
    fig.set_figheight(base_height + legend.get_window_extent().height / fig.dpi)
    fig.tight_layout()


def _add_scalebar(ax):
    """Scale bar below the axes (x in data units, y in axes units), redrawn when the x-limits change."""
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    bar, = ax.plot([], [], color='k', lw=1.5, transform=trans, clip_on=False)
    label = ax.text(0, -0.1, '', transform=trans, ha='center', va='top', fontsize='small')

    def update(ax):
        x0, x1 = ax.get_xlim()
        n = nice_length(x1 - x0)
        xs = x0 + 0.01 * (x1 - x0)
        bar.set_data([xs, xs + n], [-0.05, -0.05])
        label.set_position((xs + n / 2, -0.1))
        label.set_text(f'{n:g} trial' + ('' if n == 1 else 's'))

    ax.callbacks.connect('xlim_changed', update)


def example_sessions(trials, home, n_blocks, min_trials, after_day, distinct_best=False, min_block_reward=None):
    """Candidate example sessions of one cohort. Returns (sessions, blocks):
    sessions: one row per candidate (animal, session), best first, with its number of blocks, away blocks,
    trials, shortest block, day, lowest block reward rate and away_block (position of the away block);
    blocks: one row per block of the cohort (env, trials, reward rate, port of the highest reward %, ...).

    trials: the cohort's trial table; home: its environment ('str' or 'unstr'). A candidate has n_blocks blocks,
    exactly one of them away (env != home), every block at least min_trials trials long, and day > after_day.
    distinct_best: also, consecutive blocks never have their highest reward % at the same port.
    min_block_reward: also, the rat's reward rate is above this in every block.
    Order: sessions whose away block is in the middle (not first or last) first, then the shortest."""
    blocks = trials.groupby(['animal', 'session', 'block_num']).agg(
        env=('env', 'first'), n=('trial', 'size'), day=('day_task', 'first'), group=('block_group', 'first'),
        profile=('rewprobfull', 'first'), reward_rate=('reward', 'mean'))
    blocks['away'] = blocks.env != home
    blocks['best_port'] = blocks.profile.map(lambda p: np.argmax(parse_profile(p)) + 1 if isinstance(p, str)
                                             else np.nan)
    blocks['same_best'] = blocks.groupby(['animal', 'session']).best_port.diff().eq(0)

    sessions = blocks.groupby(['animal', 'session']).agg(
        blocks=('n', 'size'), away=('away', 'sum'), trials=('n', 'sum'), shortest=('n', 'min'),
        day=('day', 'first'), same_best=('same_best', 'any'), min_reward_rate=('reward_rate', 'min'))
    sessions['away_block'] = blocks[blocks.away].reset_index('block_num').group.groupby(['animal', 'session']).first()

    keep = ((sessions.blocks == n_blocks) & (sessions.away == 1) & (sessions.day > after_day)
            & (sessions.shortest >= min_trials))
    if distinct_best:
        keep &= ~sessions.same_best
    if min_block_reward is not None:
        keep &= sessions.min_reward_rate > min_block_reward
    sessions = sessions[keep].copy()
    sessions['middle'] = sessions.away_block.between(2, n_blocks - 1)
    return sessions.sort_values(['middle', 'trials'], ascending=[False, True]), blocks

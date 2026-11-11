from utils.opconNosepokeFunctions import *
from utils.plotSettings import *
animal = 'Vulpix'
box = 8
date = '2025-07-01'
path = f"L:/4portProb/Box{box}/{animal}/"       # for any les animal

files = get_files(path)
set_cwd(path)
files = [file for file in files if (file >=f'{animal}-*.dat')]
files.sort()
print(files)
list_sessionMarker = [13]

from utils.trializer_v4_synced import trializer_v4_synced
sessdf = trializer_v4_synced(files, animal)
sessdf['datetime'] = pd.to_datetime(sessdf.eptime, unit='s')
SESSION_GAP_SEC = 40 * 60

# if things look wonky maybe the data is not sorted!
# (it should be though the dat files are sorted per animal)
# sessdf = sessdf.sort_values(["animal", "eptime"]).copy()
gap_sec = sessdf.groupby("animal")["eptime"].diff()
new_session = gap_sec.isna() | (gap_sec > SESSION_GAP_SEC)

# Session numbers per animal: 1, 2, 3, ...
sessdf["session"] = new_session.groupby(sessdf["animal"]).cumsum().astype(int)
sessdf.rename(columns={'trial_rewprob':'rewprob'}, inplace = True)
# if you want to add block numbers, uncomment this section
sessdf['sess_block'] = 0
thing = sessdf.groupby(['animal', 'task', 'session']).datetime.head(1).astype('datetime64[ns]').diff().dt.seconds>2400
sessdf.loc[thing[thing == True].index, 'sess_block'] = 1
sessdf['sess_block'] = sessdf.groupby(['animal', 'task']).sess_block.cumsum()

# this plots blockwise raster of choice
import ipywidgets as widgets
from IPython.display import display
from matplotlib.patches import Rectangle
from matplotlib import transforms
# import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

band_cmap = 'Greys'       # any sequential matplotlib cmap, e.g. 'Oranges'
band_range = (0.12, 0.65) # part of the cmap to use: lightest (lowest prob) -> darkest (highest prob)
rect_alpha = 1.0          # opaque bands keep the shades distinct; lower range above instead of alpha to lighten
r_tick_color = 'xkcd:red'       # rewarded: tall black tick
nr_tick_color = 'xkcd:blue'   # unrewarded: short grey tick
rew_lw, unrew_lw = 1.0, 1.0 # lw
rew_h, unrew_h = 0.2, 0.2   # half-heights of the ticks
scalebar_frac = 0.1   # scale bar is the largest 1/2/5 x 10^k trials that fits in this fraction of the visible x-range

# one shade per reward probability, evenly spaced along the cmap so neighbouring probs are easy to tell apart
rewprob_vals = sorted(sessdf['rewprob'].dropna().unique())
shades = plt.get_cmap(band_cmap)(np.linspace(*band_range, len(rewprob_vals)))
rew_palette = {p: c for p, c in zip(rewprob_vals, shades)}

def nice_length(span, frac=scalebar_frac):
    # largest 1, 2 or 5 x 10^k that is <= frac * span
    target = max(span * frac, 1)
    mag = 10 ** np.floor(np.log10(target))
    for m in (5, 2, 1):
        if m * mag <= target:
            return m * mag
    return mag

# Create a function to update the plot based on user input
def update_plot(session_range):
    sessdf_filtered = sessdf[sessdf['block_num'].between(*session_range)].copy()
    fig, ax = plt.subplots(figsize=(14, 3))
    # tick height encodes outcome, all centred on the port
    rew = sessdf_filtered[sessdf_filtered['reward'] == 1]
    unrew = sessdf_filtered[sessdf_filtered['reward'] == 0]
    ax.vlines(rew['trial'], rew['port']-rew_h, rew['port']+rew_h, color=r_tick_color, lw=rew_lw, zorder=3)
    ax.vlines(unrew['trial'], unrew['port']-unrew_h, unrew['port']+unrew_h, color=nr_tick_color, lw=unrew_lw, zorder=3)

    last = sessdf.groupby('block_num')['trial'].max().to_list()
    start = sessdf.groupby('block_num')['trial'].min().to_list()
    # per-block list of reward probs for ports 1-4
    block_probs = [[float(j) for j in i.strip('[]').split()] for i in sessdf.groupby('block_num').head(1).rewprobfull]
    for ind, l in enumerate(last):
        # background bands, slightly shorter than 1 so each port row has a white gap
        for port, p in enumerate(block_probs[ind], start=1):
            if p in rew_palette:
                ax.add_patch(Rectangle((start[ind]-0.5, port-0.4), l-start[ind]+1, 0.8,
                                       fc=rew_palette[p], alpha=rect_alpha, ec='none', zorder=0))
        ax.axvline(l+0.5, color='white', lw=1, zorder=1)

    ax.set_ylim(0.5, 4.5)
    ax.invert_yaxis()   # port 1 on top
    ax.set_yticks([1, 2, 3, 4], ['Port 1', 'Port 2', 'Port 3', 'Port 4'])
    ax.tick_params(left=False)
    ax.set_xticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    # scale bar instead of an x-axis: x in data units, y in axes units, so it stays
    # below the plot and is redrawn to a sensible length whenever you zoom or pan
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    bar, = ax.plot([], [], color='k', lw=1.5, transform=trans, clip_on=False)
    bar_label = ax.text(0, -0.1, '', transform=trans, ha='center', va='top', fontsize='small')

    def update_scalebar(ax):
        x0, x1 = ax.get_xlim()
        n = nice_length(x1 - x0)
        xs = x0 + 0.01 * (x1 - x0)
        bar.set_data([xs, xs + n], [-0.05, -0.05])
        bar_label.set_position((xs + n / 2, -0.1))
        bar_label.set_text(f'{n:g} trial' + ('' if n == 1 else 's'))

    ax.callbacks.connect('xlim_changed', update_scalebar)
    x_limit = (sessdf_filtered['trial'].min() - 0.5, sessdf_filtered['trial'].max() + 0.5)
    ax.set_xlim(*x_limit)   # triggers update_scalebar

    # single legend row above the plot: bands match the background, lines match the ticks
    handles = [Patch(fc=c, alpha=rect_alpha, ec='none', label=f'{p:g}%') for p, c in rew_palette.items()]
    outcome_handles = [Line2D([], [], color=r_tick_color, lw=rew_lw*1.5, label='Rewarded'),
                       Line2D([], [], color=nr_tick_color, lw=unrew_lw*1.5, label='Unrewarded')]
    ax.legend(handles=handles + outcome_handles, ncol=len(handles) + 2, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), frameon=False, handlelength=1.2, columnspacing=1)
    plt.tight_layout()
    plt.show()

# Define the session range using a widget
session_range_slider = widgets.IntRangeSlider(
    value=[250, 300],
    min=sessdf['block_num'].min(),
    max=sessdf['block_num'].max(),
    step=1,
    description='Block Range:',
    continuous_update=False
)

# Create an interactive plot using interactive
interactive_plot = widgets.interactive(update_plot, session_range=session_range_slider)

# Display the interactive plot
display(interactive_plot)
"""Matplotlib style, colours and labels of the paper figures, and save_figure.

Importing this module sets the style for every figure (seaborn 'ticks', Inter font if installed, no top/right
spines, grey tick labels, editable text in .svg and .pdf).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from utils.config import DATA_DIR, FIG_DIR, REPO

FONT_PATH = 'C:/Windows/Fonts/Inter-Light.ttf'  # matplotlib's default font is used if this file is missing
PARULA_PATH = REPO / 'utils' / 'parula_colors.csv'  # MATLAB's parula colour map, 256 rows of r, g, b

# ---- style ----
sns.set_theme()
sns.set_style('ticks')
sns.set_context('notebook')

if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_manager.FontProperties(fname=FONT_PATH).get_name()

plt.rcParams.update({
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.labelcolor': 'grey',
    'ytick.labelcolor': 'grey',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large',
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'savefig.format': 'svg',
    'savefig.transparent': True,
    'svg.fonttype': 'none',  # svg text stays text (editable), not paths; the font must be installed to view it
    'pdf.fonttype': 42,      # same for pdf: embed TrueType fonts so text stays editable
})

# ---- colours and labels ----
# rat cohorts, i.e. the environment each cohort is trained in
COHORT_STYLE = {'unstr': {'color': 'xkcd:barney purple', 'label': 'Uncorrelated'},
                'str': {'color': 'xkcd:emerald green', 'label': 'Correlated'}}
# block types (the block's own reward profile): structured (one of the four correlated profiles) or shuffled
BLOCK_STYLE = {'str': {'color': 'xkcd:dark lime green', 'label': 'Structured'},
               'unstr': {'color': 'xkcd:wisteria', 'label': 'Shuffled'}}
# blocks by cohort and block type: home blocks in the environment colour, away blocks in the block-type colour
ENV_STYLE = {'pure_unstr': {'color': COHORT_STYLE['unstr']['color'], 'label': 'Pure uncorrelated'},
             'str_in_unstr': {'color': BLOCK_STYLE['str']['color'], 'label': 'Correlated in uncorrelated'},
             'pure_str': {'color': COHORT_STYLE['str']['color'], 'label': 'Pure correlated'},
             'unstr_in_str': {'color': BLOCK_STYLE['unstr']['color'], 'label': 'Uncorrelated in correlated'}}
# trial outcome
REWARD_STYLE = {1: {'color': 'red', 'label': 'Rewarded'},
                0: {'color': 'blue', 'label': 'Unrewarded'}}
# groups within the uncorrelated cohort (utils/cohorts.py GROUPS)
GROUP_STYLE = {'blocked': {'color': COHORT_STYLE['unstr']['color'], 'label': 'Blocked'},
               'unblocked': {'color': 'xkcd:light purple', 'label': 'Unblocked'},
               'cued': {'color': 'xkcd:pale purple', 'label': 'Cued'},
               'uncued': {'color': 'xkcd:dark lavender', 'label': 'Uncued'}}


def get_parula_cmap():
    """MATLAB's parula colour map."""
    return LinearSegmentedColormap.from_list('parula', pd.read_csv(PARULA_PATH).to_numpy())


def get_corr_cmap(alpha=0.7):
    """Faded blue (-1) -> white (0) -> red (+1) map of the MATLAB corner plots (paramCorrelations.m cp_cmap:
    redwhiteblue(256, 0, [-1, 1]) mixed with white, alpha = 0 white, 1 full colour). Use with vmin=-1, vmax=1."""
    poles = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=float)  # blue, white, red
    return LinearSegmentedColormap.from_list('redwhiteblue_faded', 1 - alpha * (1 - poles), N=256)


def week_colors(n_weeks):
    """Parula colours for weeks 1 to n_weeks; the last 10% (pale yellow) is left out so it shows on white."""
    return get_parula_cmap()(np.linspace(0, 0.9, n_weeks))


def save_figure(fig, name, source_data=None):
    """Save fig as figs/output/<name>.svg and .png, and the plotted numbers as figs/intermediate/<name>.csv.
    source_data: a DataFrame, or {cohort: DataFrame}, stacked with a 'cohort' index level."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.svg')
    fig.savefig(FIG_DIR / f'{name}.png', dpi=300)
    if source_data is not None:
        if isinstance(source_data, dict):
            source_data = pd.concat(source_data, names=['cohort'])
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        source_data.to_csv(DATA_DIR / f'{name}.csv')

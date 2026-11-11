"""Heatmaps of 4 x 4 choice transition matrices."""
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

from utils.config import PORTS
from utils.plotSettings import get_parula_cmap


def plot_transition_matrix(ax, tm, title, vmin, vmax=1):
    """Rows: port at trial t, columns: port at trial t+1 (notebook cell 19). Log colour scale, so rare
    switches and the frequent stays (the diagonal) are both visible."""
    sns.heatmap(tm, ax=ax, cmap=get_parula_cmap(), norm=LogNorm(vmin=vmin, vmax=vmax), square=True,
                xticklabels=PORTS, yticklabels=PORTS, cbar_kws={'label': 'Transition probability'})
    _label(ax, title)


def plot_difference_matrix(ax, diff, title, lim=None):
    """Rewarded minus unrewarded transition matrix (notebook cell 21); red = more likely after a reward."""
    lim = np.nanmax(np.abs(diff)) if lim is None else lim
    sns.heatmap(diff, ax=ax, cmap='seismic', center=0, vmin=-lim, vmax=lim, square=True,
                xticklabels=PORTS, yticklabels=PORTS, cbar_kws={'label': 'Rewarded − unrewarded'})
    _label(ax, title)


def _label(ax, title):
    ax.set(xlabel='Choice at trial t+1', ylabel='Choice at trial t', title=title)
    ax.tick_params(length=0)
    ax.tick_params(axis='y', rotation=0)

"""Heatmaps of 4 x 4 choice transition matrices (rows: port at trial t, columns: port at trial t+1)."""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

from utils.config import PORTS
from utils.plotSettings import get_parula_cmap


def port_matrix_frame(matrix):
    """A port x port matrix as a DataFrame (index 'port at t', columns ports at t+1), for the source data."""
    return pd.DataFrame(matrix, index=pd.Index(PORTS, name='port at t'), columns=PORTS)


def plot_transition_matrix(ax, tm, title, vmin, vmax=1):
    """Transition probabilities on a log colour scale, so rare switches and the frequent stays (the diagonal)
    are both visible."""
    sns.heatmap(tm, ax=ax, cmap=get_parula_cmap(), norm=LogNorm(vmin=vmin, vmax=vmax), square=True,
                xticklabels=PORTS, yticklabels=PORTS)
    _label(ax, title, 'Transition probability')


def plot_difference_matrix(ax, diff, title, lim=None):
    """Rewarded minus unrewarded transition matrix; red = more likely after a reward. lim: colour scale from
    -lim to lim (default: the largest absolute difference)."""
    lim = np.nanmax(np.abs(diff)) if lim is None else lim
    sns.heatmap(diff, ax=ax, cmap='seismic', center=0, vmin=-lim, vmax=lim, square=True,
                xticklabels=PORTS, yticklabels=PORTS)
    _label(ax, title, 'Rewarded − unrewarded')


def _label(ax, title, colorbar_label):
    """Axis labels and title; the colour-bar label reads top to bottom, facing the colour scale."""
    ax.set(xlabel='Choice at trial t+1', ylabel='Choice at trial t', title=title)
    ax.tick_params(length=0)
    ax.tick_params(axis='y', rotation=0)
    ax.collections[0].colorbar.ax.set_ylabel(colorbar_label, rotation=-90, va='bottom')

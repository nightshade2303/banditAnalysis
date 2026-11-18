# Correlation between the reward % of each pair of ports, computed from the task's design, not from the data
# (utils/task_profiles.py). Colour map: get_corr_cmap in utils/plotSettings.py (blue -1, white 0, red +1).
#   a. environments (as in figs/rMAB_figure_1.ipynb):
#      correlated env.: the 4 structured profiles (8/9 of blocks) and the 32 shuffled ones (1/9), weighted by use;
#      uncorrelated env.: every outcome of the task's generator (Gaussian profile around a random best port,
#      ports scrambled by random swaps), equally weighted;
#   b. block types: the 4 structured profiles; the 32 shuffled profiles; each profile equally weighted.
# Run cell by cell, or from the repo root: python -m figs.env_correlations

# %% Settings
FIGSIZE = (7, 3.2)
CMAP_ALPHA = 0.7           # strength of the colours (0 white, 1 full colour)
ANNOTATE = False           # write each coefficient in its cell
CBAR_TICKS = [-1, 0, 1]
CBAR_LABEL_SIZE = 'small'

# %% Helpers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.config import PORTS
from utils.plotSettings import BLOCK_STYLE, COHORT_STYLE, get_corr_cmap, save_figure
from utils.task_profiles import (STR_PROFILES, UNSTR_PROFILES, correlated_env_profiles, port_correlation,
                                 uncorrelated_env_profiles)


def port_frame(matrix):
    return pd.DataFrame(matrix, index=pd.Index(PORTS, name='port'), columns=PORTS)


def corr_heatmap(ax, matrix, title, color):
    """Square heatmap from -1 to 1 with its colour bar in an inset as tall as the matrix."""
    cax = ax.inset_axes([1.05, 0, 0.06, 1])
    sns.heatmap(matrix, ax=ax, cbar_ax=cax, cmap=get_corr_cmap(CMAP_ALPHA), vmin=-1, vmax=1, square=True,
                annot=ANNOTATE, fmt='.2f', xticklabels=PORTS, yticklabels=PORTS)
    cax.set_yticks(CBAR_TICKS)
    cax.set_ylabel('Correlation coefficient', rotation=-90, va='bottom', fontsize=CBAR_LABEL_SIZE)
    ax.set(xlabel='Port', ylabel='Port')
    ax.set_title(title, color=color)
    ax.tick_params(length=0)
    ax.tick_params(axis='y', rotation=0)


def correlation_figure(matrices, styles, title_suffix, name):
    """One heatmap per entry of matrices ({key: matrix}, left to right), titled and coloured by styles[key]."""
    fig, axes = plt.subplots(1, len(matrices), figsize=FIGSIZE)
    for ax, (key, matrix) in zip(axes, matrices.items()):
        corr_heatmap(ax, matrix, styles[key]['label'] + title_suffix, styles[key]['color'])
    fig.tight_layout()
    save_figure(fig, name, source_data=matrices)
    for key, matrix in matrices.items():
        print(f"{styles[key]['label']}{title_suffix}:\n{matrix.round(3).to_string()}")


# %% a: correlated and uncorrelated environments
correlation_figure({'str': port_frame(port_correlation(*correlated_env_profiles())),
                    'unstr': port_frame(port_correlation(uncorrelated_env_profiles()))},
                   COHORT_STYLE, ' env.', 'env_correlations')

# %% b: structured and shuffled blocks
correlation_figure({'str': port_frame(port_correlation(STR_PROFILES)),
                    'unstr': port_frame(port_correlation(UNSTR_PROFILES))},
                   BLOCK_STYLE, ' blocks', 'block_type_correlations')

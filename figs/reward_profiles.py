# Example reward profiles (reward % at ports 1-4) as bar plots, from figs/rMAB_figure_1.ipynb (bar plots cell):
# the 4 structured profiles (top two rows) and 4 of the 32 shuffled profiles (bottom two rows). Profiles are the
# task's, not computed from the data.
# Run cell by cell, or from the repo root: python -m figs.reward_profiles

# %% Plot (profiles and the examples shown: utils/task_profiles.py)
import matplotlib.pyplot as plt
import pandas as pd

from utils.plotSettings import BLOCK_STYLE, save_figure
from utils.task_profiles import PORTS, STR_PROFILES, UNSTR_EXAMPLES, UNSTR_PROFILES

profiles = {'str': STR_PROFILES, 'unstr': UNSTR_PROFILES[UNSTR_EXAMPLES]}
source = pd.concat({b: pd.DataFrame(p, columns=[f'port_{i}' for i in PORTS]) for b, p in profiles.items()},
                   names=['block_type', 'example'])
fig, axes = plt.subplots(4, 2, figsize=(6, 7), sharex=True)
for row0, (block_type, probs) in zip([0, 2], profiles.items()):  # structured rows 1-2, shuffled rows 3-4
    for i, p in enumerate(probs):
        r, c = divmod(i, 2)
        ax = axes[row0 + r, c]
        ax.bar(PORTS, p, width=1, color=BLOCK_STYLE[block_type]['color'])
        ax.set(ylim=(0, 80), yticks=[0, 80], xticks=PORTS, ylabel='Reward %')
for ax in axes[-1]:
    ax.set_xlabel('Ports')
fig.tight_layout()
save_figure(fig, 'reward_profiles', source_data=source)

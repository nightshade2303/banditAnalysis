# Reward rate across trials, correlated (hsbdf) vs uncorrelated (sessdf) rats, expert data.
#   1. every block, the two cohorts;
#   2. structured blocks only, in the correlated vs the uncorrelated environment;
#   3. shuffled blocks only, likewise;
#   4. from block 1 into block 2 of the same session.
# Each is the mean +- SEM across rats of each rat's mean over its blocks, smoothed (utils/config.py). Dashed
# lines: reward rate of random choices (utils/chance_levels.py), black where the curves share one chance level.
# Run cell by cell, or from the repo root: python -m figs.reward_rate

# %% Settings
LEGEND_FIRST = ['Correlated rats', 'Correlated env.']   # legend order: correlated first
# blocks of each cohort (= environment) by block type: key -> (environment, block type)
ENV_BLOCKS = {'pure_str': ('str', 'str'), 'str_in_unstr': ('unstr', 'str'),
              'unstr_in_str': ('str', 'unstr'), 'pure_unstr': ('unstr', 'unstr')}
BLOCK_TYPE_FIGURES = {'structured_blocks': 'str', 'shuffled_blocks': 'unstr'}  # output name -> block type

# %% Load both cohorts
from utils.chance_levels import CHANCE_REWARD
from utils.config import BLOCK_1_TO_2
from utils.load_data import load_cohorts
from utils.plot_curves import block_segments_figure, every_block_figure, plot_chance
from utils.plotSettings import BLOCK_STYLE, COHORT_STYLE, save_figure
from utils.trial_curves import block_trial_matrix, segment_matrix, smooth_curves

cohorts = load_cohorts()
rat_styles = {task: {'color': s['color'], 'label': f"{s['label']} rats"} for task, s in COHORT_STYLE.items()}
env_styles = {key: {'color': COHORT_STYLE[env]['color'], 'label': f"{COHORT_STYLE[env]['label']} env."}
              for key, (env, _) in ENV_BLOCKS.items()}

# %% Figure 1: reward rate over the first trials of every block, the two cohorts
curves = {task: smooth_curves(block_trial_matrix(df, 'reward')) for task, df in cohorts.items()}

fig, ax = every_block_figure(curves, rat_styles, 'Reward rate', LEGEND_FIRST)
plot_chance(ax, [(CHANCE_REWARD[task], rat_styles[task]['color']) for task in curves], label=True)
fig.tight_layout()
save_figure(fig, 'reward_rate_every_block', source_data=curves)

# %% Figures 2-3: one block type (structured or shuffled), in the correlated vs the uncorrelated environment
for name, block_type in BLOCK_TYPE_FIGURES.items():
    keys = [key for key, (_, bt) in ENV_BLOCKS.items() if bt == block_type]
    blocks = {key: cohorts[ENV_BLOCKS[key][0]].query('env == @block_type') for key in keys}
    curves = {key: smooth_curves(block_trial_matrix(df, 'reward')) for key, df in blocks.items()}

    fig, ax = every_block_figure(curves, env_styles, 'Reward rate', LEGEND_FIRST, legend_loc='center right')
    plot_chance(ax, [(CHANCE_REWARD[key], env_styles[key]['color']) for key in keys])
    ax.set_title(f"{BLOCK_STYLE[block_type]['label']} blocks", color=BLOCK_STYLE[block_type]['color'])
    fig.tight_layout()
    save_figure(fig, f'reward_rate_{name}', source_data=curves)
    for key, df in blocks.items():
        print(f"{key}: {len(curves[key])} rats, {df.groupby(['animal', 'block_num']).ngroups} blocks")

# %% Figure 4: reward rate from block 1 into block 2 of the same session
curves = {task: smooth_curves(segment_matrix(df, 'reward', BLOCK_1_TO_2), BLOCK_1_TO_2)
          for task, df in cohorts.items()}

fig, ax = block_segments_figure(curves, rat_styles, BLOCK_1_TO_2, 'Reward rate', LEGEND_FIRST,
                                legend_loc='lower right')
plot_chance(ax, [(CHANCE_REWARD[task], rat_styles[task]['color']) for task in curves])
fig.tight_layout()
save_figure(fig, 'reward_rate_block1_to_block2', source_data=curves)

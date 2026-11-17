# Choice raster of one rat's sessions (rodentBanditAnalysis.ipynb, choice raster cell), from the same cleaned
# trial tables as the paper figures: sessdf for uncorrelated rats, hsbdf for correlated rats. All days, no
# minimum session or block length; only sessions dropped by the loaders (wrongly rewarded or duplicated trials)
# are missing. Bands: each port's reward probability in the block, shaded in the rat's dataset colour, or, with
# COLOR_BY_ENV, in the colour of each block's own reward profile (BLOCK_STYLE: structured dark lime green,
# shuffled wisteria).
# Ticks: rewarded tall and black, unrewarded short and grey.
# Run cell by cell, or from the repo root: python -m figs.raster_generator

# %% Settings
ANIMAL = 'Vulpix'
SESSIONS = (40, 40)   # first and last session to plot (inclusive), session numbers of the loaded table
COLOR_BY_ENV = False
FIG_WIDTH = 7         # inches
# example sessions per cohort, (rat, session); None picks one (see the example cell)
EXAMPLES = {'unstr': None, 'str': None}

# %% Load the rat's trials
import matplotlib.pyplot as plt
import pandas as pd

from utils.cohorts import COHORTS, UNBLOCKED
from utils.config import EXPERT_AFTER_DAY, N_TRIALS
from utils.load_data import load_cohorts, load_hsbdf, load_sessdf
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.raster import fit_legend, plot_raster

if ANIMAL in COHORTS['str']:
    df, color = load_hsbdf(animals=[ANIMAL]), COHORT_STYLE['str']['color']
elif ANIMAL in COHORTS['unstr'] + UNBLOCKED:
    df, color = load_sessdf(animals=[ANIMAL]), COHORT_STYLE['unstr']['color']
else:
    raise ValueError(f"{ANIMAL} is not in COHORTS['str'], COHORTS['unstr'] or UNBLOCKED (utils/cohorts.py)")
print(f'{ANIMAL}: {df.session.nunique()} sessions ({df.session.min()} to {df.session.max()}), {len(df)} trials')


def raster_figure(rows, color, color_by_env, env_style=None):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.2))
    legend = plot_raster(ax, rows, color, color_by_env=color_by_env, env_style=env_style)
    fit_legend(fig, legend, 2.2)
    return fig


def source(rows):
    return rows.set_index(['session', 'block_num', 'trial'])[['port', 'reward', 'rewprobfull', 'env']]


# %% Static figure of the chosen sessions
rows = df[df.session.between(*SESSIONS)]
fig = raster_figure(rows, color, COLOR_BY_ENV)
name = f'raster_{ANIMAL}_sessions_{SESSIONS[0]}-{SESSIONS[1]}' + ('_by_env' if COLOR_BY_ENV else '')
save_figure(fig, name, source_data=source(rows))
print(f'{name}: {len(rows)} trials, {rows.block_num.nunique()} blocks')

# %% Example session for each cohort: 4 blocks, 3 in the cohort's own (home) environment and 1 in the other (away)
# environment, each block coloured by its block type. Candidates: expert sessions (day > EXPERT_AFTER_DAY) whose 4 blocks all
# have at least N_TRIALS trials; the away block in the middle of the session (block 2 or 3) first, then the
# shortest sessions. All candidates are saved in figs/intermediate/raster_example_candidates.csv.
candidates = []
for home, cohort_df in load_cohorts(after_day=0, min_blocks=None).items():
    blocks = cohort_df.groupby(['animal', 'session', 'block_num']).agg(
        env=('env', 'first'), n=('trial', 'size'), day=('day_task', 'first'), group=('block_group', 'first'))
    blocks['away'] = blocks.env != home
    s = blocks.groupby(['animal', 'session']).agg(blocks=('n', 'size'), away=('away', 'sum'), trials=('n', 'sum'),
                                                  shortest=('n', 'min'), day=('day', 'first'))
    s['away_block'] = blocks[blocks.away].reset_index('block_num').group.groupby(['animal', 'session']).first()
    s = s[(s.blocks == 4) & (s.away == 1) & (s.day > EXPERT_AFTER_DAY) & (s.shortest >= N_TRIALS)].copy()
    s['middle'] = s.away_block.isin([2, 3])
    s = s.sort_values(['middle', 'trials'], ascending=[False, True]).assign(home=home)
    candidates.append(s)
    print(f'{home}: {len(s)} candidate sessions from {s.index.get_level_values(0).nunique()} rats')

    rat, session = EXAMPLES[home] or s.index[0]
    rows = cohort_df[(cohort_df.animal == rat) & (cohort_df.session == session)]
    fig = raster_figure(rows, COHORT_STYLE[home]['color'], color_by_env=True)  # blocks in BLOCK_STYLE colours
    name = f'raster_example_{home}_{rat}_session_{session}'
    save_figure(fig, name, source_data=source(rows))
    print(f'  {name}: {len(rows)} trials, away block = block {int(s.loc[(rat, session), "away_block"])}')
pd.concat(candidates).to_csv('figs/intermediate/raster_example_candidates.csv')

# %% Interactive (notebook or interactive window only): browse the rat's sessions with a slider
try:
    from IPython import get_ipython
    interactive = get_ipython() is not None
except ImportError:
    interactive = False
if interactive:
    import ipywidgets as widgets
    from IPython.display import display

    def show(sessions):
        raster_figure(df[df.session.between(*sessions)], color, COLOR_BY_ENV)
        plt.show()

    slider = widgets.IntRangeSlider(value=SESSIONS, min=df.session.min(), max=df.session.max(), step=1,
                                    description='Sessions:', continuous_update=False)
    display(widgets.interactive(show, sessions=slider))

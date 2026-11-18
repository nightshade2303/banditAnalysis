# Choice raster: a tick per trial at the chosen port (rewarded tall and black, unrewarded short and grey) over
# bands whose shade gives each port's reward % in the block (utils/raster.py). Data: the cleaned trial tables of
# the paper figures (sessdf for uncorrelated rats, hsbdf for correlated rats), all days, no minimum session or
# block length.
#   a. sessions SESSIONS of rat ANIMAL, bands in the rat's cohort colour or, with COLOR_BY_ENV, in the colour of
#      each block's own profile (BLOCK_STYLE: structured dark lime green, shuffled wisteria);
#   b. one example session per cohort: EXAMPLE_BLOCKS blocks, all but one in the cohort's own (home)
#      environment, coloured by block type (see utils/raster.py example_sessions for how it is chosen);
#   c. (notebook / interactive window only) a slider to browse the rat's sessions.
# Not run by make_all. Run cell by cell, or from the repo root: python -m figs.raster_generator

# %% Settings
ANIMAL = 'Vulpix'
SESSIONS = (40, 40)   # first and last session to plot (inclusive), session numbers of the loaded table
COLOR_BY_ENV = False
FIG_WIDTH = 7         # inches
RASTER_HEIGHT = 2.2   # inches, without the legend
# example sessions per cohort, (rat, session); None picks the best candidate
EXAMPLES = {'unstr': None, 'str': None}
EXAMPLE_BLOCKS = 4
# extra rules for the example sessions of these cohorts: the best (80%) port differs between consecutive blocks,
# and the rat's reward rate is above MIN_BLOCK_REWARD in every block
FILTERED_COHORTS = ['unstr']
MIN_BLOCK_REWARD = 0.5

# %% Load the rat's trials
import matplotlib.pyplot as plt
import pandas as pd

from utils.cohorts import COHORTS, UNBLOCKED
from utils.config import DATA_DIR, EXPERT_AFTER_DAY, N_TRIALS
from utils.load_data import load_cohorts, load_hsbdf, load_sessdf
from utils.plotSettings import COHORT_STYLE, save_figure
from utils.raster import example_sessions, fit_legend, plot_raster

if ANIMAL in COHORTS['str']:
    df, color = load_hsbdf(animals=[ANIMAL]), COHORT_STYLE['str']['color']
elif ANIMAL in COHORTS['unstr'] + UNBLOCKED:
    df, color = load_sessdf(animals=[ANIMAL]), COHORT_STYLE['unstr']['color']
else:
    raise ValueError(f"{ANIMAL} is not in COHORTS['str'], COHORTS['unstr'] or UNBLOCKED (utils/cohorts.py)")
print(f'{ANIMAL}: {df.session.nunique()} sessions ({df.session.min()} to {df.session.max()}), {len(df)} trials')


def raster_figure(rows, color, color_by_env):
    """Figure with the raster of `rows`, FIG_WIDTH wide and RASTER_HEIGHT high plus the legend."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, RASTER_HEIGHT))
    legend = plot_raster(ax, rows, color, color_by_env=color_by_env)
    fit_legend(fig, legend, RASTER_HEIGHT)
    return fig


def source(rows):
    """The plotted trials, for the source-data csv."""
    return rows.set_index(['session', 'block_num', 'trial'])[['port', 'reward', 'rewprobfull', 'env']]


# %% a: the chosen sessions of ANIMAL
rows = df[df.session.between(*SESSIONS)]
fig = raster_figure(rows, color, COLOR_BY_ENV)
name = f'raster_{ANIMAL}_sessions_{SESSIONS[0]}-{SESSIONS[1]}' + ('_by_env' if COLOR_BY_ENV else '')
save_figure(fig, name, source_data=source(rows))
print(f'{name}: {len(rows)} trials, {rows.block_num.nunique()} blocks')

# %% b: example session for each cohort, expert data, every block at least N_TRIALS trials long.
# All candidates are saved in figs/intermediate/raster_example_candidates.csv.
candidates = []
for home, cohort_df in load_cohorts(after_day=0, min_blocks=None).items():
    filtered = home in FILTERED_COHORTS
    sessions, blocks = example_sessions(cohort_df, home, EXAMPLE_BLOCKS, N_TRIALS, EXPERT_AFTER_DAY,
                                        distinct_best=filtered,
                                        min_block_reward=MIN_BLOCK_REWARD if filtered else None)
    candidates.append(sessions.assign(home=home))
    print(f'{home}: {len(sessions)} candidate sessions from {sessions.index.get_level_values(0).nunique()} rats')

    rat, session = EXAMPLES[home] or sessions.index[0]
    rows = cohort_df[(cohort_df.animal == rat) & (cohort_df.session == session)]
    fig = raster_figure(rows, COHORT_STYLE[home]['color'], color_by_env=True)
    name = f'raster_example_{home}_{rat}_session_{session}'
    save_figure(fig, name, source_data=source(rows))
    print(f'  {name}: {len(rows)} trials, away block = block {int(sessions.loc[(rat, session), "away_block"])}')
    print(blocks.loc[(rat, session), ['env', 'n', 'best_port', 'reward_rate']].round(2).to_string())
pd.concat(candidates).to_csv(DATA_DIR / 'raster_example_candidates.csv')

# %% c: interactive (notebook or interactive window only): browse the rat's sessions with a slider
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

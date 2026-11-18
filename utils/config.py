"""Paths and analysis settings for the paper figures in figs/. Point the paths at your copy of the data."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# data
SESSDF_PATH = Path('R:/behavioralData/processed/sessdf-2026-09-25.csv')  # uncorrelated rats
HSBDF_PATH = Path('V:/processed/hsbdf-2026-11-11.csv')                     # correlated rats
RULES_PATH = REPO / 'utils' / 'task_data.json'                             # lesion and implant dates
FIG_DIR = REPO / 'figs' / 'output'                                         # figures
DATA_DIR = REPO / 'figs' / 'intermediate'                                  # numbers behind each figure (csv)

# which trials are analysed
EXPERT_AFTER_DAY = 30   # expert data: day_task > 30
N_TRIALS = 100          # blocks with fewer trials are dropped; longer blocks contribute their first 100
# expert plots keep only rats with more than MIN_BLOCKS blocks of at least N_TRIALS trials (None: keep all rats)
MIN_BLOCKS = None
MAX_WEEK = 8           # week-wise plots: weeks 1 to MAX_WEEK of training
MAX_DUPLICATES = 10   # sessions with more duplicated trial rows are dropped
# block 1 -> block 2 plots, in x-axis order: (block of the session, 'first' or 'last' trials, number of trials)
BLOCK_1_TO_2 = [(1, 'first', 100), (1, 'last', 25), (2, 'first', 100)]

# trial-by-trial plots: x label and ticks of the first N_TRIALS trials of a block
TRIAL_XLABEL = 'Trials in block'
TRIAL_TICKS = [1, 25, 50, 75, 100]

# moving average over trials, applied to each rat's curve before averaging across rats
SMOOTH = True
SMOOTH_WINDOW = 5               # odd, so the window is centred
SMOOTH_WITHIN_SEGMENTS = True   # block 1 -> 2 plots: don't average across the axis break or the block switch
SMOOTH_FULL_WINDOWS = True      # blank the trials whose window runs past the start or end of the curve (or segment),
                                # e.g. trials 1-2 and 99-100 for a 5-trial window, instead of averaging fewer trials

# transition matrices: how the rats of a cohort are combined
TM_AGG = 'mean'                 # 'mean' or 'median'

PORTS = [1, 2, 3, 4]

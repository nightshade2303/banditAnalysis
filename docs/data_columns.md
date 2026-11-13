# Data and measures behind the paper figures

This describes the trial tables made by [utils/load_data.py](../utils/load_data.py) and how each plotted measure is computed. The figure scripts are listed in the [README](../README.md); paths and settings are in [utils/config.py](../utils/config.py).

## Cohorts

| Cohort | `task` | Rats ([utils/cohorts.py](../utils/cohorts.py)) | Source |
|---|---|---|---|
| Uncorrelated environment | `unstr` | cued and uncued rats | sessdf csv, made by [trializer_v4_synced](trializer_v4_synced.md) |
| Correlated environment | `str` | HSB_F_006 to HSB_F_014 | hsbdf csv (`HSBDF_PATH`), same format as sessdf |

## Sessions and blocks

- `block_num`: block number from the source file, continuous across sessions.
- `session`: consecutive blocks whose start times are at most 40 min apart, split also at any gap of more than 40 min between consecutive trials, even inside a block (`add_block_groups` in [utils/dfLoading.py](../utils/dfLoading.py)), for both cohorts. For sessdf this replaces the csv's own `session` column (a 40-min gap between trials), which gives the same sessions except one.
- `block_group`: position of the block in its session; 1 is the first block.
- `day_task`: 24-hour periods since the rat's first trial, starting at 1. Expert data are trials with `day_task > 30`.

## Cleaning

1. sessdf: only task `unstr` and the cohort's rats are read; blocks with malformed reward probabilities are dropped (`read_df`).
2. hsbdf: task 12 is `str`; sessions after an implant (dates in [utils/task_data.json](../utils/task_data.json)) get task `ph_implant` and are not analysed.
3. Whole sessions with any wrongly rewarded trial are dropped (`rewarded_by_wrong_port`), in both cohorts. The rat experienced those rewards, so the rest of the session is affected too.
4. Sessions with more than 10 duplicated trial rows are dropped, in both cohorts (`remove_sessions_with_duplicate_trials`).

## Columns of the trial tables

| Column | Meaning |
|---|---|
| `animal` | Rat |
| `task` | `unstr` (uncorrelated cohort), `str` (correlated cohort), `ph_implant` (hsb after the implant) |
| `env` | Whether the block's reward profile is one of the four structured profiles (`str`) or not (`unstr`). This is a property of the block, not the cohort |
| `day_task`, `week_task` | Day and week of training (24-hour periods since the rat's first trial) |
| `datetime` | Trial start |
| `session`, `block_num`, `block_group` | See above |
| `trial` | Running trial number, increasing in time within a rat; not comparable between cohorts |
| `port` | Chosen port, 1 to 4 |
| `reward` | 1 rewarded, 0 not rewarded, NaN no outcome logged |
| `rewprob` | Reward probability (%) the box used for this trial |
| `rewprobfull` | Reward probabilities (%) of ports 1 to 4 in this block |
| `choice_t1` | Port chosen on the next trial of the same session; NaN on the session's last trial |
| `disp` | `choice_t1 - port` |
| `shift_t0` | 1 if the next choice is a different port (switch), 0 if it is the same port (stay), NaN on the session's last trial |

## Trials analysed

- Rats: the expert plots use only rats whose expert data have more than 100 blocks of at least 100 trials (`MIN_BLOCKS`; set it to `None` to keep all rats).

- Every-block plots and transition matrices: the first 100 trials of every block with at least 100 trials; shorter blocks are dropped.
- Block 1 to block 2 plots: sessions where block 1 has at least 125 trials and block 2 at least 100. The plots show the first 100 and last 25 trials of block 1 and the first 100 trials of block 2.
- First-block plots (6 to 8): `block_group == 1`, first 100 trials of blocks with at least 100 trials.
- Week-wise plots: all rats and all days (no expert cutoff, no minimum number of blocks), first block of each session, weeks 1 to 8 (`MAX_WEEK`). Week = `week_task`, 7-day periods since the rat's first trial of the task.
- Group comparisons: uncorrelated environment only. Blocked = cued + uncued rats; unblocked rats (`UNBLOCKED` in utils/cohorts.py) mostly have one block per session. Same expert cutoff, cleaning and minimum number of blocks as the other expert plots.

## Measures

- Reward rate: mean of `reward` at each trial.
- Switch rate: mean of `shift_t0` at each trial.
- Switch distance: `|disp|` on switch trials (`switch_dist`, NaN when the rat stayed), averaged over the switches at each trial.
- P(switch | outcome): mean of `shift_t0` over the trials with that outcome; the trial-wise version does this at each trial.
- P(switch to distance d | outcome): fraction of the trials with that outcome after which the next choice is d ports away.
- Switch distance after an outcome: in each block, the mean `|disp|` of the switches after that outcome; then the mean over blocks.
- Transition matrix: for each rat, P(port at trial t+1 | port at trial t), each row summing to 1; then the mean across rats (`TM_AGG = 'median'` in config.py for the median). The rewarded minus unrewarded matrix is computed per rat from rewarded and unrewarded trials, then averaged.

## Averaging and smoothing

Each rat's curve is the mean over its blocks (over its sessions for the block 1 to block 2 plots), smoothed with a centred 5-trial moving average that skips NaN values (`SMOOTH`, `SMOOTH_WINDOW`). In the block 1 to block 2 plots each segment is smoothed separately, so the average never crosses the axis break or the block switch. Figures show the mean ± SEM across rats.

## Chance levels

- Reward rate: the reward rate of random choices, i.e. the mean probability of the four ports, averaged with equal weight over the reward profiles the blocks can have (`CHANCE_REWARD` in [utils/chance_levels.py](../utils/chance_levels.py)). Every block uses the values (80, 46, 15, 10) or (80, 46, 46, 15) in some port order. Uncorrelated task: mean over the two value sets, 0.4225. Correlated blocks: mean over the four correlated profiles, 0.4225. Uncorrelated blocks given to correlated rats: mean over the other 32 port orders, 0.4056.
- Switch distance: mean distance from the starting port to the other three ports (2 from ports 1 and 4, 4/3 from ports 2 and 3), averaged over the ports the rats switched from.
- P(switch to distance d): the rat's switch rate times the fraction of the other three ports that are d ports from the starting port.

## Source data

Each figure in `figs/output/` has a csv of the same name in `figs/intermediate/` with the per-rat values that were plotted.

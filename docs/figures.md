# Paper figures

What each figure shows, which data go into it and how every number is computed.

Every figure is made by a short script in [figs/](../figs/), runnable cell by cell or with
`python -m figs.<script>`. `python -m figs.make_all` runs them all, reading the data once; it skips the raster
script. Each script writes:

- `figs/output/<name>.svg`, with editable text;
- `figs/output/<name>.png`, at 300 dpi;
- `figs/intermediate/<name>.csv`, the numbers plotted.

The thumbnails below link to `figs/output/`. That folder is not tracked by git, so the thumbnails show only once
the figures have been made locally (VS Code's Markdown preview shows them).
[data_columns.md](data_columns.md) describes the trial tables and the cleaning.

---

## Contents

| | Topic | Figures |
|---|---|---|
| 1 | [Task schematics](#1-task-schematics) | reward profiles · environment proportions · environment and block-type correlations · choice rasters |
| 2 | [Reward rate](#2-reward-rate) | every block · structured blocks · shuffled blocks · block 1 → 2 |
| 3 | [Switch rate](#3-switch-rate) | every block · block 1 → 2 · block 1 vs 2 |
| 4 | [Switch distance](#4-switch-distance) | every block · block 1 → 2 · block 1 vs 2 |
| 5 | [Switching and outcome](#5-switching-and-outcome) | P(switch) by outcome · switch distance by outcome |
| 6 | [Where rats switch to](#6-where-rats-switch-to) | P(switch) at each distance |
| 7 | [Transition matrices](#7-transition-matrices) | transitions · rewarded − unrewarded |
| 8 | [Learning over weeks](#8-learning-over-weeks) | weeks 1–8 · rats and sessions per week |
| 9 | [Groups in the uncorrelated environment](#9-groups-in-the-uncorrelated-environment) | blocked vs unblocked · cued vs uncued |
| 10 | [Sessions per week](#10-sessions-per-week) | training · expert · total expert sessions |
| 11 | [Cohort statistics](#11-cohort-statistics-first-300-trials) | first 300 trials, permutation tests |

---

## Conventions

These hold for every figure unless its card says otherwise.

**Cohorts and environments.** There are two groups of rats ([utils/cohorts.py](../utils/cohorts.py)):

| Cohort | Environment | Rats in the expert figures |
|---|---|---|
| <b style="color:#028f1e">■</b> Correlated (`str`, hsbdf) | 4 structured reward profiles, plus some shuffled blocks | 9 |
| <b style="color:#a00498">■</b> Uncorrelated (`unstr`, sessdf) | 32 shuffled profiles, plus some structured blocks | 13 |

**Block types.** A block's own reward profile is either:
- <b style="color:#7ebd01">■</b> structured (dark lime green), one of the four correlated profiles; or
- <b style="color:#a87dc2">■</b> shuffled (wisteria), any other.

The `env` column records the block type ([utils/task_profiles.py](../utils/task_profiles.py)).

**Outcome colours.** <b style="color:red">■</b> rewarded trials, <b style="color:blue">■</b> unrewarded trials.

**Expert data.** Expert figures use:
- trials after day 30 of training (`EXPERT_AFTER_DAY`);
- every rat of the cohort lists in [utils/cohorts.py](../utils/cohorts.py) that has expert data (`MIN_BLOCKS = None`;
  a number n keeps only rats with more than n blocks of at least `N_TRIALS` trials).

All settings are in [utils/config.py](../utils/config.py).

**Trials in a block.** Trial-by-trial curves use:
- blocks with at least 100 trials;
- their first 100 trials.

**Across rats.** Each rat's curve or value is computed first. Plots show the mean ± SEM across rats.

**Smoothing.** Each rat's curve is smoothed before averaging, with a centred 5-trial moving average that ignores
NaN (`SMOOTH_WINDOW`). Only complete windows are used (`SMOOTH_FULL_WINDOWS`): the first and last 2 trials of a
curve have no value, so a 100-trial curve is drawn from trial 3 to trial 98. Block 1 → 2 plots are smoothed within
each segment, never across the break or the block switch, so each segment loses 2 trials at both ends.

**Block 1 → 2.** `BLOCK_1_TO_2` lays three segments side by side:
- trials 1–100 of block 1;
- the last 25 trials of block 1;
- trials 1–100 of block 2.

A session counts only if block 1 has at least 125 trials and block 2 at least 100. Each rat's curve is the mean
over its sessions. A slanted break marks the skipped middle of block 1, and a dashed grey line marks the block
switch.

**Chance levels** ([utils/chance_levels.py](../utils/chance_levels.py)). Every block uses one of two value sets,
(80, 46, 15, 10) or (80, 46, 46, 15), in some port order.

- *Reward rate.* Chance is the reward rate of random choices: the mean of a profile's four reward %, averaged
  with equal weight over the profiles the blocks can use.

  | Blocks | Chance |
  |---|---|
  | Uncorrelated cohort (two value sets) | 0.4225 |
  | Structured blocks (4 profiles) | 0.4225 |
  | Shuffled blocks of the correlated rats (32 profiles, 22 of them from the lower set) | 0.4056 |

  Where the curves of a figure share one chance level, it is drawn as a single black dashed line. Otherwise
  each line is drawn in its curve's colour.
- *Switch distance.* Chance is the mean |distance| when the next port is drawn at random from the other three:
  2 from ports 1 and 4, 4/3 from ports 2 and 3. It is averaged over the ports the rat switched from.
- *P(switch to distance d).* Chance keeps the rat's own switch rate and draws the next port at random from the
  other three.

---

## 1. Task schematics

### Reward profiles
<img src="../figs/output/reward_profiles.png" width="360">

| | |
|---|---|
| **Script** | [figs/reward_profiles.py](../figs/reward_profiles.py) |
| **Output** | `reward_profiles` |
| **Data** | none: the task's profiles ([utils/task_profiles.py](../utils/task_profiles.py)) |

**What it shows.** Reward % at ports 1–4:
- top two rows: the 4 structured profiles, in dark lime;
- bottom two rows: 4 of the 32 shuffled profiles, in wisteria (`UNSTR_EXAMPLES`).

### Environment proportions
<img src="../figs/output/env_proportions.png" width="240">

| | |
|---|---|
| **Script** | [figs/env_proportions.py](../figs/env_proportions.py) |
| **Output** | `env_proportions` |
| **Data** | none: shares set by hand (`SHARES`) |

**What it shows.** An alluvial plot of how each environment's blocks split by block type:
- correlated environment: 0.89 structured, 0.11 shuffled;
- uncorrelated environment: 0.08 structured, 0.92 shuffled.

Each band's width is that proportion, read off the "Proportion of blocks" scale on the right of each bar (0 at the
bottom, 1 at the top, unlabelled ticks every 0.2). On the left, each block type is a rounded box, labelled above,
holding three example profiles as bar plots of reward probability per port (y axis 0–0.8), shaded as in the
rasters (darker = higher reward probability). The dots stand for the profiles not shown.

**Settings.** `SHARES`, `N_EXAMPLES`, `PROFILE_YTICKS` and the layout constants (`GAP`, `WIDTH`, `PROFILE_W`,
`YAXIS_W`, …).

### Environment correlations
<img src="../figs/output/env_correlations.png" width="420">

| | |
|---|---|
| **Script** | [figs/env_correlations.py](../figs/env_correlations.py) |
| **Output** | `env_correlations` |
| **Data** | none: computed from the task design ([utils/task_profiles.py](../utils/task_profiles.py)) |

**What it shows.** For each pair of ports, the Pearson correlation between their reward % across the blocks an
environment can produce. Blue is −1, white 0 and red +1 (`get_corr_cmap` in
[utils/plotSettings.py](../utils/plotSettings.py)). In the correlated environment, neighbouring ports go up and down
together (about +0.34), and distant ports are anticorrelated (about −0.74 to −0.80). In the uncorrelated
environment, every pair of different ports is weakly negative (about −0.27 to −0.31). That is close to −1/3, the value for
four fixed rewards shuffled uniformly among the ports.

**How it's computed.**
- *Correlated env.:* every structured and shuffled profile. The 4 structured profiles share 8/9 of the weight
  (`CORRELATED_STR_SHARE`) and the 32 shuffled ones share 1/9. The correlation is computed from the weighted
  covariance (`np.cov(..., aweights=...)`).
- *Uncorrelated env.:* every outcome of the task's generator, equally weighted. The generator picks a best port
  (1–4) and gives port x the reward ⌊(0.7·exp(−(x − best)²/(2·0.875²)) + 0.1)·100⌋ %. It then swaps the port at each
  position i = 1–4 with a random position, so there are 4 × 4⁴ outcomes.
- Both matrices are the same as the ones in `figs/rMAB_figure_1.ipynb`.

**Why the uncorrelated environment isn't 0.** Each block hands out a fixed set of rewards (80, 46, 15, 10 or 80, 46,
46, 15). The ports only trade them, so when one port gets a high reward the others must share the lower ones.
- If the four values always summed to the same total and every port order were equally likely, every pair of
  ports would correlate at exactly −1/(4 − 1) = −1/3.
- The two value sets sum to different totals (151 and 187), which weakens that link. With uniformly shuffled
  ports, every pair would be −0.29.
- The task's swap shuffle is not uniform: of the 256 swap sequences, some port orders come out 8 times and others
  15 times. That moves the pairs between −0.27 and −0.31.

### Block-type correlations
<img src="../figs/output/block_type_correlations.png" width="420">

| | |
|---|---|
| **Script** | [figs/env_correlations.py](../figs/env_correlations.py), cell b |
| **Output** | `block_type_correlations` |
| **Data** | none: computed from the task design ([utils/task_profiles.py](../utils/task_profiles.py)) |

**What it shows.** The same port × port correlation as above, for each block type on its own (each profile
equally weighted):
- *Structured blocks* (the 4 structured profiles): neighbouring ports +0.44, ports 2 and 3 ≈ 0, distant ports
  −0.81 to −0.87.
- *Shuffled blocks* (the 32 shuffled profiles): every pair is negative.
  - Neighbouring ports are the most negative (−0.38 for ports 1 & 2 and 3 & 4, −0.34 for ports 2 & 3).
  - Distant ports are −0.22 to −0.24.
  - These are the orders left once the 4 structured profiles, where neighbours share high rewards, are taken
    out, so neighbours lean towards being dissimilar.

**Settings.** `CMAP_ALPHA` (colour strength), `ANNOTATE` (write each coefficient in its cell), `CBAR_TICKS`.

### Choice rasters
<img src="../figs/output/raster_example_unstr_Vanillish_session_84.png" width="520">

| | |
|---|---|
| **Script** | [figs/raster_generator.py](../figs/raster_generator.py), not run by `make_all` |
| **Output** | `raster_<rat>_sessions_<a>-<b>`, `raster_example_<cohort>_<rat>_session_<n>`, `raster_example_candidates.csv` |
| **Data** | cleaned trial tables, all days |

**What it shows.**
- One tick per trial at the chosen port: rewarded trials tall and black, unrewarded trials short and grey.
- Background bands give each port's reward % in the block, from light (10 %) to dark (80 %).
- White vertical lines separate blocks, and a scale bar gives the number of trials.

**Example sessions.** One per cohort, chosen by `example_sessions` in [utils/raster.py](../utils/raster.py):
- expert data;
- 4 blocks of at least 100 trials;
- exactly one block of the other block type (the "away" block), preferably in the middle;
- the shortest such session.

For the uncorrelated cohort, two more rules apply:
- consecutive blocks never have their 80 % at the same port;
- the rat's reward rate is above 0.5 in every block.

Bands are coloured by block type.

**Settings.** `ANIMAL`, `SESSIONS`, `COLOR_BY_ENV`, `EXAMPLES` (to pin a session), `FILTERED_COHORTS`,
`MIN_BLOCK_REWARD`.

---

## 2. Reward rate

Script: [figs/reward_rate.py](../figs/reward_rate.py). Data: expert data, the `reward` column (no-outcome trials
left out).

### Every block
<img src="../figs/output/reward_rate_every_block.png" width="340">

| | |
|---|---|
| **Output** | `reward_rate_every_block` |
| **Unit** | rats: 9 correlated, 13 uncorrelated |
| **Shown** | mean ± SEM across rats; black dashed line: chance (0.4225, the same for both cohorts) |

**What it shows.** The reward rate over the first 100 trials of every block, correlated vs uncorrelated rats.

**How it's computed.**
1. Take every block with at least 100 trials.
2. For each trial position, average over the rat's blocks.
3. Smooth each rat's curve, then average across rats.

### Structured blocks · Shuffled blocks
<img src="../figs/output/reward_rate_structured_blocks.png" width="300"> <img src="../figs/output/reward_rate_shuffled_blocks.png" width="300">

| | |
|---|---|
| **Output** | `reward_rate_structured_blocks`, `reward_rate_shuffled_blocks` |
| **Unit** | rats: 9 correlated, 13 uncorrelated |
| **Shown** | mean ± SEM across rats; dashed: chance |

**What it shows.** One block type, as experienced in the correlated environment (green) and the uncorrelated
environment (purple):
- structured blocks are the correlated rats' home blocks but rare in the uncorrelated environment;
- shuffled blocks are the reverse.

The heading colour gives the block type.

**How it's computed.** As for every block, keeping only the blocks of that type:

| Figure | Correlated rats' blocks | Uncorrelated rats' blocks |
|---|---|---|
| Structured | 8,539 | 915 |
| Shuffled | 1,084 | 7,016 |

**Chance.**
- Structured blocks: one black line at 0.4225.
- Shuffled blocks: the chance levels differ (0.4056 in the correlated environment, 0.4225 in the uncorrelated
  one), so each is a dashed line in its environment's colour.

### Block 1 → block 2
<img src="../figs/output/reward_rate_block1_to_block2.png" width="480">

| | |
|---|---|
| **Output** | `reward_rate_block1_to_block2` |
| **Shown** | mean ± SEM across rats; black dashed line: chance |

**What it shows.** The reward rate through the first block of a session and into the second. It shows how fast
the rats recover after the reward profile changes. Segments are described under [Conventions](#conventions).

---

## 3. Switch rate

Script: [figs/switch_rate.py](../figs/switch_rate.py). A trial is a **switch** if the next choice in the session
is a different port (`shift_t0`: 1 switch, 0 stay, NaN on the session's last trial).

| Output | What it shows |
|---|---|
| `switch_rate_every_block`<br><img src="../figs/output/switch_rate_every_block.png" width="260"> | Switch rate over the first 100 trials of every block, mean ± SEM across rats |
| `switch_rate_block1_to_block2`<br><img src="../figs/output/switch_rate_block1_to_block2.png" width="340"> | The same from block 1 into block 2 |
| `switch_rate_block1_vs_block2`<br><img src="../figs/output/switch_rate_block1_vs_block2.png" width="220"> | Per rat, mean switch rate over trials 1–100 of block 1 and of block 2 (sessions of the 1 → 2 plot, not smoothed). Bars: mean ± SEM; dots: rats. In each block the cohorts are compared with a two-sided Mann-Whitney U test, p above the bars; the test table is printed. |

---

## 4. Switch distance

Script: [figs/switch_distance.py](../figs/switch_distance.py). The **switch distance** is the number of ports
between the current and the next choice, on switch trials only (`switch_dist`; NaN on stays). It runs from 1 to
3.

| Output | What it shows |
|---|---|
| `switch_distance_every_block`<br><img src="../figs/output/switch_distance_every_block.png" width="260"> | Switch distance over the first 100 trials of every block |
| `switch_distance_block1_to_block2`<br><img src="../figs/output/switch_distance_block1_to_block2.png" width="340"> | The same from block 1 into block 2 |
| `switch_distance_block1_vs_block2`<br><img src="../figs/output/switch_distance_block1_vs_block2.png" width="220"> | Per rat, mean over trials 1–100 of block 1 and of block 2: each trial's mean over the rat's sessions, then the mean over trials. Mann-Whitney U per block, as for switch rate. |

---

## 5. Switching and outcome

Data: the **first block of each session** only (blocks with at least 100 trials, their first 100 trials), expert
data. Bars: mean ± SEM across rats; dots: rats.

### P(switch) after rewarded and unrewarded trials
<img src="../figs/output/switch_prob_by_reward.png" width="240"> <img src="../figs/output/switch_prob_by_reward_trialwise.png" width="420">

| | |
|---|---|
| **Script** | [figs/switch_by_reward.py](../figs/switch_by_reward.py) |
| **Outputs** | `switch_prob_by_reward` (per rat), `switch_prob_by_reward_trialwise` (curves) |

**How it's computed.**
- *Per rat:* P(switch | outcome) is the mean of `shift_t0` over the rat's trials with that outcome.
- *Trialwise:* at each trial of block 1, the mean of `shift_t0` over the rat's blocks, using only trials with that
  outcome. That gives P(switch | outcome) at that trial. Each curve is smoothed, with one panel per outcome.

### Switch distance after rewarded and unrewarded trials
<img src="../figs/output/switch_distance_by_reward.png" width="240"> <img src="../figs/output/switch_distance_by_reward_trialwise.png" width="420">

| | |
|---|---|
| **Script** | [figs/switch_distance_by_reward.py](../figs/switch_distance_by_reward.py) |
| **Outputs** | `switch_distance_by_reward`, `switch_distance_by_reward_trialwise` |

**How it's computed.**
- *Per rat:* the mean |distance| of the switches after each outcome in each block, then the mean over blocks.
- *Chance:* a grey dashed line on each bar, the chance distance averaged over the ports the rat switched from.
- *Trialwise:* as for P(switch), using the switch distance.

---

## 6. Where rats switch to

Script: [figs/switch_by_distance.py](../figs/switch_by_distance.py). Data: first block of each session, as in §5.
Log y axis, because the larger distances are rare.

| Output | What it shows |
|---|---|
| `switch_prob_by_distance`<br><img src="../figs/output/switch_prob_by_distance.png" width="360"> | P(next choice at distance 1, 2, 3) after rewarded (red) and unrewarded (blue) trials, mean ± SEM across rats, black dashed line: chance. One panel per cohort. |
| `switch_prob_by_distance_any_outcome`<br><img src="../figs/output/switch_prob_by_distance_any_outcome.png" width="360"> | The same whatever the outcome, with distance 0 = P(stay). It has no chance line, because chance keeps the rat's own stay rate. Bars in the cohort colour, grey dots for rats (jittered). Each panel has its own y axis and legend. |

**How it's computed.** Per rat, P(distance d) = the fraction of all trials whose next choice is d ports away.
Over d = 0–3 the probabilities sum to 1. Over d = 1–3 (the by-outcome figure) they sum to the rat's switch rate
after that outcome.

---

## 7. Transition matrices

Script: [figs/transition_matrices.py](../figs/transition_matrices.py). Data: first 100 trials of every block
with at least 100 trials, expert data.

| Output | What it shows |
|---|---|
| `transition_matrices`<br><img src="../figs/output/transition_matrices.png" width="400"> | P(port at trial t+1 \| port at trial t). Rows: the current port; columns: the next. Log colour scale (parula) from 0.01 (`VMIN`) to 1, so the frequent stays (the diagonal) and the rare long switches both show. |
| `transition_matrices_rewarded_minus_unrewarded`<br><img src="../figs/output/transition_matrices_rewarded_minus_unrewarded.png" width="400"> | The transition matrix after rewarded trials minus that after unrewarded trials. Red: more likely after a reward. The colour scale is symmetric and shared by both cohorts. |

**How it's computed.**
1. For each rat, count the port → next-port pairs within sessions and divide each row by its total. A port the
   rat never chose gives a NaN row.
2. Combine the rats' matrices with `TM_AGG` (mean by default, or median), ignoring NaN.

The difference is computed per rat (rewarded − unrewarded), then combined.

---

## 8. Learning over weeks

Script: [figs/weekwise.py](../figs/weekwise.py). Data:
- **all rats and all days**, weeks 1–8 of training (`MAX_WEEK`);
- the first block of each session;
- its first 100 trials (blocks with at least 100 trials).

| Output | What it shows |
|---|---|
| `weekwise_unstr`, `weekwise_str`<br><img src="../figs/output/weekwise_unstr.png" width="460"> | Reward rate, switch rate and switch distance over the first 100 trials, one curve per week (parula colours, early = dark blue). Mean ± SEM across the rats that ran that week. |
| `weekwise_counts`<br><img src="../figs/output/weekwise_counts.png" width="360"> | Number of sessions and of rats behind each week's curves. |

The rats change from week to week, so a difference between weeks mixes learning with who was running.

---

## 9. Groups in the uncorrelated environment

Script: [figs/group_comparisons.py](../figs/group_comparisons.py). Two comparisons, all expert data:
- **blocked vs unblocked** rats: the uncorrelated cohort vs rats without block changes, whose sessions are mostly
  a single block;
- **cued vs uncued** blocked rats: block changes signalled or not.

Every figure here is named `<measure>_<comparison>_<selection>`:
- `<measure>`: `reward_rate`, `switch_rate` or `switch_distance`;
- `<comparison>`: `blocked_vs_unblocked` or `cued_vs_uncued`;
- `<selection>`: `first_block` (first block of the session only), `every_block` or, for cued vs uncued,
  `block1_to_block2`.

<img src="../figs/output/switch_rate_cued_vs_uncued_block1_to_block2.png" width="440">

| Output | What it shows |
|---|---|
| `…_first_block`, `…_every_block` | Curves over the first 100 trials, mean ± SEM across rats, as in §2–4 |
| `…_cued_vs_uncued_block1_to_block2` | Block 1 → 2, as in §2–4 |
| `switch_rate_cued_vs_uncued_early_block2`<br><img src="../figs/output/switch_rate_cued_vs_uncued_early_block2.png" width="180"> | Per rat, the switch rate over the first 10 trials of block 2 (`EARLY_TRIALS`, not smoothed). Bars: mean ± SEM; dots: rats. Mann-Whitney U test in the title. |

---

## 10. Sessions per week

Script: [figs/sessions_per_week.py](../figs/sessions_per_week.py). Three datasets:
- **UU**: uncorrelated, unblocked;
- **UB**: uncorrelated, blocked;
- **SB**: correlated (structured), blocked.

Sessions are counted after cleaning, whatever their blocks. A rat's sessions per week = its sessions divided by
the weeks in which it ran at least one session, so weeks with no sessions (e.g. after an implant) do not count.
Over each dataset's bars, lines give the mean (solid), median (dashed) and, for expert data, minimum (dotted);
their values are written above.

| Output | What it shows |
|---|---|
| `sessions_by_week_training`<br><img src="../figs/output/sessions_by_week_training.png" width="420"> | Weeks 1–8, all rats: sessions each rat ran in each week, one panel per week. Each rat keeps the same position in every panel. |
| `sessions_per_week_training`, `sessions_per_week_expert`<br><img src="../figs/output/sessions_per_week_expert.png" width="420"> | Sessions per week of each rat, pooled over weeks 1–8 or over the expert data (rats of the expert figures). |
| `sessions_total_expert`<br><img src="../figs/output/sessions_total_expert.png" width="420"> | Total expert sessions per rat, all rats. Rats left out of the expert figures (too few long blocks) are hatched; rats with no expert session show 0. |

---

## 11. Cohort statistics: first 300 trials

<img src="../figs/output/cohort_first_300_trials.png" width="440">

| | |
|---|---|
| **Script** | [figs/cohort_first_300.py](../figs/cohort_first_300.py) |
| **Output** | `cohort_first_300_trials` and `…_tests.csv`. Saving is commented out in the script; uncomment it to write the files. |
| **Unit** | rats: 13 uncorrelated, 9 correlated |
| **Test** | unpaired permutation test, 10,000 permutations ([stats/permutation_tests.py](../stats/permutation_tests.py)) |

**What it shows.** Reward rate, switch rate and switch distance over the first 300 trials of each session,
compared between cohorts. Boxes show the median and quartiles in the cohort colour, whiskers reach 1.5 × IQR,
and the dots are rats.

**How it's computed.**
1. Within each session, sort the trials in time and keep the first 300. These cross block switches, so they
   cover about block 1 and part of block 2.
2. Sessions with fewer than 300 trials are dropped: 8 % of uncorrelated and 4 % of correlated sessions.
3. For each rat, take the mean over each session's 300 trials, then the mean over its sessions.

**The test**, a port of the lab's `permTest_2samples.m`:
1. The statistic is |mean₁ − mean₂|.
2. The null pools all rats and splits them at random into groups of the original sizes, 10,000 times.
3. p is the fraction of null statistics strictly greater than the observed one. It can be 0, which is shown as
   p < 0.0001.

**Settings.** `N_FIRST`, `N_PERMS`, `Y_MIN`, `BOX_ALPHA`, `DOT_SIZE`, `JITTER`.

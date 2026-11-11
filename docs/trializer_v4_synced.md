# trializer_v4_synced

This document explains how [utils/trializer_v4_synced.py](../utils/trializer_v4_synced.py) turns an animal's raw `.dat` files into a trial table (`sessdf`): which rows become trials, which trials are left out, and how each column is computed. It describes the code as of 2026-09-25.

## Overview

```python
from trializer_v4_synced import trializer_v4_synced
sessdf = trializer_v4_synced(files, animal)
```

- `files`: list of `.dat` file names for one animal, in chronological order. The working directory must be the animal's folder.
- `animal`: the animal's name. `'Torchic'` gets an extra repair step (see step 3).
- `nosepoke_events` (optional, default `((21,), (20,), (17, 19), (37, 18))`): the poke codes of each port (1 to 4) used to locate the triggering poke for the lick window. Same mapping as `dict_port_rel`.

The module imports `fix_torchic_data` and `opconNosepokeFunctions` as top-level names, so `utils/` must be on the Python path.

One row of the output is one trial. A trial starts at a trial-start event (`81`) and runs until the next trial start.

## Event codes used

| Code | Meaning | Value column holds |
|---|---|---|
| 81 | Trial start | Port of the trial (1 to 4) |
| 51 / 86 | Rewarded / unrewarded outcome | — |
| 88 | Reward probability the box used for this trial | Probability in % |
| 83 | Block reward probability for one port | Port. **The probability is in the `smtime` column** |
| 61 / 62 | Block start / block end | — |
| 111 | Block end marker used in Uxie's task | — |
| 21, 20, 17/19, 37/18 | Nosepoke at port 1, 2, 3, 4 | 1 = poke in, 0 = poke out |
| 28, 25, 24, 22 | Lick sensor (which one depends on the rig and date) | 1 = lick on, 0 = lick off |

Port-to-poke-code mapping (`dict_port_rel`): port 1 = 21, port 2 = 20, port 3 = 17 or 19, port 4 = 37 or 18.

Only rows from the task state machines (`sm` 12 and 13) are used.

## Processing steps

### 1) Load the files

Each file is read with `merge_files_to_df` and tagged with a session id (`sess_id` = its position in `files`). All sessions are then concatenated into one array, so later steps can work across the whole dataset but still tell sessions apart.

### 2) Keep the useful rows

Rows are kept if the event is one of the codes above and the state machine is 12 or 13. Everything else (other state machines, camera events, firmware error messages written into the log) is dropped.

### 3) Torchic repair

For Torchic only, `fix_torchic_rows` replaces each run of three `81` rows with a single `81` row carrying the missing port (the one of 1 to 4 that was not logged). See [utils/fix_torchic_data.py](../utils/fix_torchic_data.py).

### 4) Resolve duplicated trial starts

Some rigs, during some periods, log two or three `81` rows at the same instant (same `smtime` and `eptime`) for one trial. Only one of them is the port the animal chose. This step keeps that one and removes the others.

**How the chosen row is identified.** Only from the hardware poke that triggered the trial:

1. Find the **triggering pokes**: poke-in events (a poke code with value 1) that are
   - in the same session as the `81` rows,
   - after the previous trial start, and
   - at most 50 ms before the `81` rows.

   In the data checked so far the triggering poke is almost always 1 ms before the `81`.
2. Convert each triggering poke to its port with `dict_port_rel`.
3. Keep the `81` row whose port matches a triggering poke.

**Which trials are kept or left out.**

| Situation | Result |
|---|---|
| Exactly one `81` row matches a triggering poke | That row is kept, the others are removed. The trial is kept. |
| No triggering poke in the window | The trial is left out |
| Several triggering pokes matching several `81` rows | The trial is left out |
| A matching row cannot be found for any other reason | The trial is left out |

Left-out trials keep one `81` row during processing so that the neighbouring trials' windows stay the same, and are removed from the output at the very end.

The number of removed extra rows is stored per trial in `n_dup81` (0 for normal trials).

Normal trials (a single `81` row) are not changed by this step.

### 5) Choose the lick marker for each session

The lick sensor code differs between rigs and has changed over time for some animals, for example 22 in some of an animal's sessions and 25 in others. The marker is therefore chosen **per session**:

1. If the session contains `28`, use 28; otherwise `25`; otherwise `24`; otherwise `22`.
2. Only lick-on rows (value 1) count as licks.
3. For marker 22 only: on some rigs 22 is wired to the same input as the nosepokes, so a nosepoke also produces a 22 row at the same time. A 22 row directly before or after a poke row with the same `smtime` (same session) is not counted as a lick.

### 6) Define trials

Every remaining `81` row starts a trial. The trial runs until the next `81` row (or the end of the data). From the `81` row:

- `port` = its value
- `task` = its state machine (12 or 13)
- `eptime` = its `eptime` (Unix time in seconds)

### 7) Blocks (`block_num`)

1. Block starts are the `61` rows. Block ends are the `62` rows, or the `111` rows if the data contains any `111`.
2. If the total numbers of block starts and block ends are equal, a trial's block is the number of block ends before it.
3. Otherwise, a trial's block is the number of block starts before it, minus 1. This fallback was added for Togepi's data, where some block ends are missing.

The choice between 2 and 3 is made once for the **whole dataset**, not per session. A single session stopped mid-block (one more `61` than `62`) switches every session to the fallback.

`block_num` counts continuously across sessions. Block markers in sessions without trials still count, so block numbers can skip.

### 8) Poke times

- `poke_in_time`: `smtime` of the last poke event at the trial's port before its `81`.
- `poke_out_time`: `smtime` of the first poke event at the trial's port after its `81` (normally the poke-out).

Both use all poke codes of the port from `dict_port_rel`.

### 9) Lick times

1. **Window start: the triggering poke**, taken as the last poke-in (value 1) at the trial's port before its `81`, using any of the port's codes in `nosepoke_events`. If there is no such poke (for example the first trial), the window starts at the `81`.
2. **Window end: the next `81`**.
3. `first_lick_time` = first valid lick after the window start. `last_lick_time` = last valid lick before the window end. Valid licks are the ones from step 5.
4. A trial with no valid lick in its window gets NaN for both.

### 10) Outcome (`reward`)

Within the trial's window, 1 if a `51` comes before any `86`, 0 if an `86` comes first, NaN if neither occurs.

### 11) Reward probability (`trial_rewprob`)

The value of the first `88` row in the trial's window. **This is the probability the box actually used to decide the reward.** In almost every animal checked, reward rates match `trial_rewprob`, including in trials where it does not match the chosen port's block probability (for exceptions see Known limitations).

### 12) Block probabilities (`rewprobfull`)

The block's `83` rows sorted by port, written as a string: `'[46 15 80 10]'` means port 1 = 46%, port 2 = 15%, port 3 = 80%, port 4 = 10%. Blocks are delimited as in step 7.

### 13) Wrongly rewarded trials (`rewarded_by_wrong_port`)

In rigs that log duplicated trial starts, the box sometimes takes the reward probability from one of the extra `81` rows instead of the chosen one. The trial's choice is real, but its reward was drawn from another port's probability.

A trial is flagged when:

- it had duplicated trial starts (`n_dup81 > 0`), and
- its `trial_rewprob` (`88`) differs from the current block probability of the chosen port (the latest `83` for that port since the last block start).

If the extra port happens to have the same block probability as the chosen one, the trial is not flagged, because the reward came from the right probability. Normal trials are always `False`.

### 14) Output

The trials left out in step 4 are removed, and the table is returned. `trial` is numbered before that removal, and the index is not reset, so both have small gaps where trials were left out.

## Output columns

| Column | Meaning |
|---|---|
| `animal` | Animal name |
| `task` | State machine of the trial start (12 or 13) |
| `block_num` | Block number, continuous across sessions (step 7) |
| `trial` | Trial number across all sessions (gaps where trials were left out) |
| `port` | Chosen port, 1 to 4 |
| `reward` | 1 = rewarded, 0 = not rewarded, NaN = no outcome logged |
| `trial_rewprob` | Reward probability the box used (`88`), in % |
| `rewprobfull` | Block probabilities of ports 1 to 4 |
| `poke_in_time`, `poke_out_time` | `smtime` of the poke in and poke out (ms within the session) |
| `first_lick_time`, `last_lick_time` | `smtime` of the first and last lick in the trial (ms within the session) |
| `eptime` | Unix time of the trial start (s) |
| `n_dup81` | Number of extra trial-start rows removed from this trial (0 = normal trial) |
| `rewarded_by_wrong_port` | True if the reward was drawn from another port's probability (step 13) |

## Which trials to use

| Trials | Use |
|---|---|
| `n_dup81 == 0` | Normal trials |
| `n_dup81 > 0` and `rewarded_by_wrong_port == False` | Recovered duplicated trials: choice and reward are both valid |
| `rewarded_by_wrong_port == True` | Valid choice, but the reward came from the wrong probability. Exclude from reward-based analyses |

To exclude wrongly rewarded trials:

```python
sessdf = sessdf[~sessdf['rewarded_by_wrong_port']]
```

Do not filter on `n_dup81 > 0`: that also removes the recovered trials.

The animal still experienced the wrong rewards, and they influence its later choices. For analyses that depend on reward history, consider excluding whole sessions with many flagged trials.

## Known limitations

- **Block fallback is dataset-wide** (step 7). One session with mismatched block starts and ends changes the block numbering method for every session of that animal. The `- 1` in the fallback has not been checked for all animals.
- **Reward delivery problems are not detected.** In a few animals and sessions the reward rate is below the logged `88` probability (for example Hatenna, Finneon 2025-01-09, Xatu 2024-08-03 and 2024-08-29). The trializer does not flag these.
- **Torchic**: the duplicate resolution in step 4 also runs after `fix_torchic_rows`. Any `81` groups the repair leaves behind are resolved or left out by the poke rule.

## Related

- [utils/check_dat_files.py](../utils/check_dat_files.py) scans every animal's `.dat` files for the problems described here (mixed lick markers, duplicated trial starts, block marker mismatches, reward mismatches) and writes `dat_check_summary.csv` and `dat_check_files.csv`.

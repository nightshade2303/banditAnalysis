"""trializer_v4_synced

This is a standalone copy of `trializer_v4`; **only the lick block changed** (everything else — ports,
 reward, blocks, poke_in/out, rewprob, eptime — is byte-identical to the original). The three changes:

  A. ON edges only (value==1): the "last lick" is the last lick ON, not the last row (OFF).
  B. window starts at the TRIGGER POKE (last poke-ON into the trial's port before the 81),
     not at the 81 itself; the first trial (no trigger poke) anchors at the 81.
  C. first/last lick are bounded by the next 81; a trial with no ON lick in its window is NaN.

  More changes that were added 24-09-2026:
  Check for different lick marker per .dat file to determine the lick marker for that day.
  As a consequence, function now requires the list of files themselves, and merge_files_to_df is a part of the trializer itself.
  Finally, there are also markers added if there are duplicate 81s in the original dat file, which allow removal of those trials as invalid if they were not rewarded correctly.
"""
import pandas as pd
import numpy as np
import json
from utils.fix_torchic_data import fix_torchic_rows
from utils.opconNosepokeFunctions import merge_files_to_df

def trializer_v4_synced(files, animal, nosepoke_events=((21,), (20,), (17, 19), (37, 18))):
    # specify all useful events
    trialStartMarker = 81
    trialHitMarker = 51
    trialMissMarker = 86
    trialProbMarker = 88
    rewardProbMarker = 83
    blockStartMarker = 61
    blockEndMarker = 62
    list_sessionMarker = [12, 13]
    list_pokeMarker = [21, 20, 17, 37, 19, 18]
    list_UxieBlockEndMarker = [111]
    dict_port_rel = {
        1: [21],
        2: [20],
        3: [17, 19],
        4: [37, 18]
    }
    list_lickMarker = [28, 24, 25, 22]

    useful_events = [
        trialStartMarker, trialHitMarker, trialMissMarker,
        blockStartMarker, blockEndMarker,
        trialProbMarker, rewardProbMarker
        ] + list_pokeMarker + list_lickMarker + list_UxieBlockEndMarker

    useful_events = np.array(useful_events)

    # convert dataframe to numpy instead
        # merged data with a session id column appended (same rows/order as merge_files_to_df(files))
    sess_frames = []
    for day_id, f in enumerate(files):
        sdf = merge_files_to_df([f])
        sdf['sess_id'] = day_id
        sess_frames.append(sdf)
    data = pd.concat(sess_frames, ignore_index=True).to_numpy()

    # select useful data (correct events and state machines only)
    useful_mask = np.isin(data[:, 0], useful_events) & np.isin(data[:, 2], list_sessionMarker)
    useful_data = data[useful_mask]
    if animal == 'Torchic':
        useful_data = fix_torchic_rows(useful_data)

    # resolve duplicated trial starts (several 81s logged at the same smtime + eptime):
    # keep only the 81 whose port matches the hardware poke that triggered it, i.e. a poke-in
    # within 50 ms before the 81s, in the same session and after the previous trial start.
    # If no single 81 matches a triggering poke (no trigger, or several triggers matching several
    # 81s), the trial is ditched. Torchic's duplicates are handled by fix_torchic_rows instead.
    row_n_dup81 = np.zeros(len(useful_data), dtype=int)
    row_ditch_trial = np.zeros(len(useful_data), dtype=bool)
    
    port_of_poke_code = {code: port for port, codes in dict_port_rel.items() for code in codes}
    # numeric copies (files with firmware messages in the log are read with text columns)
    # initialize columns and find same-time-81s
    smtime = useful_data[:, 4].astype(float)
    eptime_all = useful_data[:, 5].astype(float)
    is_81 = (useful_data[:, 0] == trialStartMarker) # find 81s
    same_time_as_prev_81 = np.zeros(len(useful_data), dtype=bool)
    same_time_as_prev_81[1:] = (is_81[1:] & is_81[:-1] #  only one before and after?
                                & (smtime[1:] == smtime[:-1])
                                & (eptime_all[1:] == eptime_all[:-1]))
    dup_group_starts = np.flatnonzero(is_81 & ~same_time_as_prev_81 & np.append(same_time_as_prev_81[1:], False))
    trial_start_rows = np.flatnonzero(is_81 & ~same_time_as_prev_81)
    poke_in_rows = np.flatnonzero(np.isin(useful_data[:, 0], list_pokeMarker) & (useful_data[:, 1] == 1))

    extra_81_rows = []
    # for every extra 81
    for group_start in dup_group_starts:
        group_end = group_start + 1
        while group_end < len(useful_data) and same_time_as_prev_81[group_end]:
            group_end += 1
        group = np.arange(group_start, group_end)

        # triggering pokes: poke-ins after the previous trial start, within 50 ms, same session
        prev_trial_start = trial_start_rows[np.searchsorted(trial_start_rows, group_start) - 1] if group_start > trial_start_rows[0] else -1
        candidate_pokes = poke_in_rows[np.searchsorted(poke_in_rows, prev_trial_start, side='right'):np.searchsorted(poke_in_rows, group_start)]
        trigger_pokes = [row for row in candidate_pokes
                            if useful_data[row, -1] == useful_data[group_start, -1]
                            and 0 <= smtime[group_start] - smtime[row] <= 50]
        triggered_ports = {port_of_poke_code[useful_data[row, 0]] for row in trigger_pokes}
        matching_81s = [row for row in group if useful_data[row, 1] in triggered_ports]

        if len(matching_81s) == 1:
            kept_81 = matching_81s[0]
        else:
            kept_81 = group[0]      # keeps the trial window intact; the trial is dropped at the end
            row_ditch_trial[kept_81] = True
        row_n_dup81[kept_81] = len(group) - 1
        extra_81_rows.extend(row for row in group if row != kept_81)

    keep_rows = np.ones(len(useful_data), dtype=bool)
    keep_rows[extra_81_rows] = False
    useful_data = useful_data[keep_rows]
    row_n_dup81 = row_n_dup81[keep_rows]
    row_ditch_trial = row_ditch_trial[keep_rows]

    # subset of used data
    sess_col = useful_data[:, -1].astype(int)
    event_col = useful_data[:, 0]
    value_col = useful_data[:, 1]
    time_col = useful_data[:, 4]

    # per-session lick marker -> mask of valid lick ON rows
    # (22 rows adjacent to a same-time poke are excluded instead of deleted; deleting them
    #  in the old cell did not change the order of any other rows, so results are equivalent)
    poke_mask = np.isin(event_col, list_pokeMarker)
    same_prev = np.zeros(len(useful_data), dtype=bool)
    same_next = np.zeros(len(useful_data), dtype=bool)
    same_prev[1:] = poke_mask[:-1] & (time_col[1:] == time_col[:-1]) & (sess_col[1:] == sess_col[:-1])
    same_next[:-1] = poke_mask[1:] & (time_col[:-1] == time_col[1:]) & (sess_col[:-1] == sess_col[1:])
    adjacent_22 = (event_col == 22) & (same_prev | same_next)

    valid_lick_mask = np.zeros(len(useful_data), dtype=bool)
    sess_lick_marker = {}
    for s in np.unique(sess_col):
        in_sess = sess_col == s
        ev_s = event_col[in_sess]
        marker = next((m for m in (28, 25, 24) if np.any(ev_s == m)), 22)
        sess_lick_marker[files[s]] = marker
        m = in_sess & (event_col == marker) & (value_col == 1)
        if marker == 22:
            m &= ~adjacent_22
        valid_lick_mask |= m

    # extract trial info from useful_data, including how many trials
    trial_idx = np.where(useful_data[:, 0] == trialStartMarker)[0]

    # get trial start indices and trial ends (here, every next trial + end of the event_col)
    n_trials = len(trial_idx)
    trial_starts = trial_idx
    trial_ends = np.append(trial_idx[1:], len(event_col))

    # got ports selected
    ports = useful_data[trial_idx, 1].astype(int)

    # get task info
    task = useful_data[trial_idx, 2].astype(int)

    # get eptime info
    eptime = useful_data[trial_idx, 5].astype(int)

    # duplicated trial starts resolved above: number of extra 81s removed from each trial
    n_dup81 = row_n_dup81[trial_idx]
    ditch_trial = row_ditch_trial[trial_idx]

    # Preallocate arrays - there can only be as many trials
    poke_in_times = np.full(n_trials, np.nan)
    poke_out_times = np.full(n_trials, np.nan)
    first_lick_times = np.full(n_trials, np.nan)
    last_lick_times = np.full(n_trials, np.nan)
    outcomes = np.full(n_trials, np.nan)
    block_nums = np.zeros(n_trials, dtype=int)
    trial_rewprobs = np.full(n_trials, np.nan)

    # Vectorized block assignment
    block_start_idx = np.where(event_col == 61)[0]
    if not np.any(event_col == 111):
        block_end_idx = np.where(event_col == 62)[0]
    else:
        block_end_idx = np.where(event_col == 111)[0]

    # edited because of missing block ends in Togepi's data
    if block_start_idx.shape[0] == block_end_idx.shape[0]:
        # searchsorted finds indices in block_end_idx where trial_starts can be inserted
        # this will give chronologically sorted trials (block-wise)
        block_bins = np.searchsorted(block_end_idx, trial_starts, side='right')
        block_nums[:] = block_bins
    else:
        block_bins = np.searchsorted(block_start_idx, trial_starts, side='left')
        block_nums[:] = block_bins - 1  # dont know if the -1 will work for all rats

    # Precompute poke event indices for all possible poke events
    # for every poke event, get the index
    poke_event_indices = {poke: np.flatnonzero(event_col == poke) for poke in set(sum(dict_port_rel.values(), []))}

    # Vectorized poke in time (last poke before trial start)
    # for every poke hardware pin
    for port_num, poke_codes in dict_port_rel.items():
        # select port n trials
        trial_mask = (ports == port_num)

        # get the indices of the trial starting
        trial_starts_port = trial_starts[trial_mask]
        if trial_starts_port.size == 0:
            # sometimes there are no trials for a port
            continue

        # for every hardware pin (21/20/19/18 etc) find where the poke indices are (first poke per trial)
        poke_inds = np.concatenate([poke_event_indices[code] for code in poke_codes if code in poke_event_indices])
        if poke_inds.size == 0:
            continue
        poke_inds.sort()

        # the last poke index is found by checking if there are any trials after poke_inds
        last_poke_idx = np.searchsorted(poke_inds, trial_starts_port, side='right') - 1
        # didnt fully get what happened here
        valid = (last_poke_idx >= 0) & (last_poke_idx < poke_inds.size)
        poke_in_times_for_port = np.full(trial_mask.sum(), np.nan)
        poke_in_times_for_port[valid] = time_col[poke_inds[last_poke_idx[valid]]]
        poke_in_times[trial_mask] = poke_in_times_for_port

    # Vectorized poke out time (first poke after trial start)
    for port_num, poke_codes in dict_port_rel.items():
        trial_mask = (ports == port_num)
        trial_starts_port = trial_starts[trial_mask]
        if trial_starts_port.size == 0:
            continue
        poke_inds = np.concatenate([poke_event_indices[code] for code in poke_codes if code in poke_event_indices])
        if poke_inds.size == 0:
            continue
        poke_inds.sort()
        insert_pos = np.searchsorted(poke_inds, trial_starts_port, side='left')
        # similar logic as above, but didnt really get it
        valid = (insert_pos < poke_inds.size)
        poke_out_times_for_port = np.full(trial_mask.sum(), np.nan)
        poke_out_times_for_port[valid] = time_col[poke_inds[insert_pos[valid]]]
        poke_out_times[trial_mask] = poke_out_times_for_port

    # ── Lick block — SYNC-ENGINE SEMANTICS (the only change vs trializer_v4) ─────
    # A. ON edges only (value==1): "last lick" is the last ON, not the last row (OFF).
    # B. window starts at the TRIGGER POKE (last poke-ON into the trial's port before the
    #    81); first trial / no trigger poke -> anchor at the 81.
    # C. first/last lick bounded by the next 81; no ON lick in window -> NaN.
    nosepoke_events = list(nosepoke_events)
    trigger_pos = trial_starts.astype(np.int64).copy()   # fallback: the 81 itself
    for port_num in (1, 2, 3, 4):
        codes = np.atleast_1d(nosepoke_events[port_num - 1])   # all poke codes of this port
        trial_mask = (ports == port_num)
        ts_port = trial_starts[trial_mask]
        if ts_port.size == 0:
            continue
        on_pos = np.flatnonzero(np.isin(event_col, codes) & (value_col == 1))
        if on_pos.size == 0:
            continue
        # last poke-ON of this port's code strictly before each trial's 81
        last_on = np.searchsorted(on_pos, ts_port, side='left') - 1
        tp = ts_port.astype(np.int64).copy()             # default: the 81 (no trigger)
        ok = last_on >= 0
        tp[ok] = on_pos[last_on[ok]]
        trigger_pos[trial_mask] = tp

    lick_mask = valid_lick_mask   # ON edges only
    lick_times = time_col[lick_mask]
    lick_indices = np.flatnonzero(lick_mask)
    # first ON lick strictly AFTER the trigger poke; last ON lick strictly BEFORE next 81
    lo = np.searchsorted(lick_indices, trigger_pos, side='right')
    hi = np.searchsorted(lick_indices, trial_ends, side='left')
    has_lick = hi > lo
    first_lick_times[has_lick] = lick_times[lo[has_lick]]
    last_lick_times[has_lick] = lick_times[hi[has_lick] - 1]

    # Vectorized outcome detection
    hit_mask = (event_col == 51)
    miss_mask = (event_col == 86)
    hit_indices = np.flatnonzero(hit_mask)
    miss_indices = np.flatnonzero(miss_mask)
    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        first_hit = hit_indices[(hit_indices >= start) & (hit_indices < end)]
        first_miss = miss_indices[(miss_indices >= start) & (miss_indices < end)]
        if first_hit.size == 0 and first_miss.size == 0:
            outcomes[i] = np.nan
        elif first_hit.size == 0:
            outcomes[i] = 0
        elif first_miss.size == 0:
            outcomes[i] = 1
        else:
            outcomes[i] = int(first_hit[0] < first_miss[0])

    # Vectorized trial rewprob extraction
    prob_mask = (event_col == 88)
    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        prob_idx = np.flatnonzero(prob_mask[start:end])
        if prob_idx.size > 0:
            trial_rewprobs[i] = value_col[start:end][prob_idx[0]]

    # Vectorized rewprobfull assignment
    # Precompute reward probability string for each block
    # edited because of missing block ends (62) in Togepi's data
    block_count = max(block_nums) + 1
    block_rewprob_strs = np.empty(block_count, dtype=object)
    for block in range(block_count):
        if block in block_nums:
            # if starts and ends have the same shape.
            if block_start_idx.shape[0] == block_end_idx.shape[0]:
                if block < len(block_start_idx) and block < len(block_end_idx):
                    start = block_start_idx[block]
                    end = block_end_idx[block] + 1
                    block_data = useful_data[start:end]
                    rewprob_events = block_data[block_data[:, 0] == 83]
                    if rewprob_events.shape[0] > 0:
                        ports_in_block = rewprob_events[:, 1].astype(int)
                        values_in_block = rewprob_events[:, 4].astype(int)
                        sorted_indices = np.argsort(ports_in_block)
                        sorted_values = values_in_block[sorted_indices]
                        block_rewprob_strs[block] = str(sorted_values)
                    else:
                        block_rewprob_strs[block] = ''
                else:
                    block_rewprob_strs[block] = ''
            # otherwise every block is defined by the block starts (61)
            else:
                if block < len(block_start_idx):
                    start = block_start_idx[block]
                    end = np.append(block_start_idx, useful_data.shape[0]).astype(int)[block + 1]
                    block_data = useful_data[start:end]
                    rewprob_events = block_data[block_data[:, 0] == 83]
                    if rewprob_events.shape[0] > 0:
                        ports_in_block = rewprob_events[:, 1].astype(int)
                        values_in_block = rewprob_events[:, 4].astype(int)
                        sorted_indices = np.argsort(ports_in_block)
                        sorted_values = values_in_block[sorted_indices]
                        block_rewprob_strs[block] = str(sorted_values)
                    else:
                        block_rewprob_strs[block] = ''
                else:
                    block_rewprob_strs[block] = ''
        else:
            block_rewprob_strs[block] = ''
    rewprobfull = block_rewprob_strs[block_nums]

    # a duplicated trial was rewarded with the wrong port's probability if its 88 (the probability
    # the box actually used, = trial_rewprob) differs from the poked port's current block probability
    # (latest 83 for that port since the last block start)
    # basically recheck rp rewarded on with block prob and 88
    rewarded_by_wrong_port = np.zeros(n_trials, dtype=bool)
    block_start_rows = np.flatnonzero(event_col == blockStartMarker)
    for port_num in dict_port_rel:
        dup_trials = np.flatnonzero((n_dup81 > 0) & (ports == port_num))
        port_prob_rows = np.flatnonzero((event_col == rewardProbMarker) & (value_col == port_num))
        if dup_trials.size == 0 or port_prob_rows.size == 0:
            continue
        last_prob = np.searchsorted(port_prob_rows, trial_starts[dup_trials]) - 1
        last_block_start = np.searchsorted(block_start_rows, trial_starts[dup_trials]) - 1
        has_prob = (last_prob >= 0) & ((last_block_start < 0) |
                   (port_prob_rows[np.maximum(last_prob, 0)] > block_start_rows[np.maximum(last_block_start, 0)]))
        poked_port_prob = time_col[port_prob_rows[np.maximum(last_prob, 0)]].astype(float)
        rewarded_by_wrong_port[dup_trials] = has_prob & (trial_rewprobs[dup_trials] != poked_port_prob)

    # convert most floats to int64 to save memory
    ports = ports.astype(int)
    eptime = eptime.astype(int)

    # Assemble DataFrame (identical columns to trializer_v4)
    sessdf_vec = pd.DataFrame({
        'animal': [animal] * n_trials,
        'task': task,
        'block_num': block_nums,
        'trial': np.arange(n_trials),
        'port': ports,
        'reward': outcomes,
        'trial_rewprob': trial_rewprobs,
        'rewprobfull': rewprobfull,
        'poke_in_time': poke_in_times,
        'poke_out_time': poke_out_times,
        'first_lick_time': first_lick_times,
        'last_lick_time': last_lick_times,
        'eptime': eptime,
        'n_dup81': n_dup81,
        'rewarded_by_wrong_port': rewarded_by_wrong_port,
    })
    # drop trials whose duplicated 81s could not be matched to a single triggering poke
    return sessdf_vec[~ditch_trial]


def apply_task_rules_from_json(sessdf: pd.DataFrame, rules_path: str, add_days: bool = False) -> pd.DataFrame:
    """Same task assignment as trializer_v4.apply_task_rules_from_json. add_days=True then adds
    day_task/week_task via add_day_task(sessdf, rules_path): bias-correction days are skipped and
    the count restarts at the tasks in "reset_day_count_on" (see hsb_task_data.json)."""
    with open(rules_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_map = cfg.get("base_task_map", {})
    if base_map:
        cast_map = {}
        for k, v in base_map.items():
            try:
                cast_map[int(k)] = v
            except (TypeError, ValueError):
                cast_map[k] = v
        sessdf["task"] = sessdf["task"].replace(cast_map)

    sessdf["datetime"] = pd.to_datetime(sessdf["datetime"])
    for rule in cfg.get("rules", []):
        mask = (sessdf["datetime"] > pd.to_datetime(rule["after"])) & (sessdf["animal"] == rule["animal"])
        sessdf.loc[mask, "task"] = rule["task"]

    if add_days:
        from utils.dfLoading import add_day_task
        sessdf = add_day_task(sessdf, rules_path=rules_path)
    return sessdf

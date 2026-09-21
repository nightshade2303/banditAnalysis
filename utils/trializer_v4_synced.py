"""trializer_v4_synced

This is a standalone copy of `trializer_v4`; **only the lick block changed** (everything else — ports, reward, blocks, poke_in/out, rewprob, eptime — is byte-identical to the original). The three changes:

  A. ON edges only (value==1): the "last lick" is the last lick ON, not the last row (OFF).
  B. window starts at the TRIGGER POKE (last poke-ON into the trial's port before the 81),
     not at the 81 itself; the first trial (no trigger poke) anchors at the 81.
  C. first/last lick are bounded by the next 81; a trial with no ON lick in its window is NaN.

"""
import pandas as pd
import numpy as np
import json


def trializer_v4_synced(df, animal, nosepoke_events=(21, 20, 19, 18)):
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
    data = df.to_numpy()

    # select useful data (correct events and state machines only)
    useful_mask = np.isin(data[:, 0], useful_events) & np.isin(data[:, 2], list_sessionMarker)
    useful_data = data[useful_mask]

    # subset of used data
    event_col = useful_data[:, 0]
    value_col = useful_data[:, 1]
    time_col = useful_data[:, 4]

    # identify lick marker and process useful_data accordingly
    lick_marker = None
    if np.any(event_col == 28):
        lick_marker = 28
    elif np.any(event_col == 25):
        lick_marker = 25
    elif np.any(event_col == 24):
        lick_marker = 24
    else:
        # this currently is for any animal that
        # has GPIO responding to both lick and pokes
        lick_marker = 22

        lick_idx = np.flatnonzero(np.isin(event_col, lick_marker))
        poke_mask = np.isin(event_col, list_pokeMarker)

        adjacent_lick_mask = np.zeros(len(useful_data), dtype=bool)

        prev_idx = lick_idx - 1
        next_idx = lick_idx + 1

        valid_prev = prev_idx >= 0
        valid_next = next_idx < len(useful_data)

        # lick row is +1 index from a poke row, and column 4 matches
        adjacent_lick_mask[lick_idx[valid_prev]] |= (
            poke_mask[prev_idx[valid_prev]] &
            (time_col[lick_idx[valid_prev]] == time_col[prev_idx[valid_prev]])
        )

        # lick row is -1 index from a poke row, and column 4 matches
        adjacent_lick_mask[lick_idx[valid_next]] |= (
            poke_mask[next_idx[valid_next]] &
            (time_col[lick_idx[valid_next]] == time_col[next_idx[valid_next]])
        )
        useful_data = useful_data[~adjacent_lick_mask]
        event_col = useful_data[:, 0]
        value_col = useful_data[:, 1]
        time_col = useful_data[:, 4]

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
    if np.any(event_col) != 111:
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
        code = nosepoke_events[port_num - 1]
        trial_mask = (ports == port_num)
        ts_port = trial_starts[trial_mask]
        if ts_port.size == 0:
            continue
        on_pos = np.flatnonzero((event_col == code) & (value_col == 1))
        if on_pos.size == 0:
            continue
        # last poke-ON of this port's code strictly before each trial's 81
        last_on = np.searchsorted(on_pos, ts_port, side='left') - 1
        tp = ts_port.astype(np.int64).copy()             # default: the 81 (no trigger)
        ok = last_on >= 0
        tp[ok] = on_pos[last_on[ok]]
        trigger_pos[trial_mask] = tp

    lick_mask = (event_col == lick_marker) & (value_col == 1)   # ON edges only
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
    })
    return sessdf_vec

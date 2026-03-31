import numpy as np
import pandas as pd

def extract_trials_vectorized(useful_data, trial_idx, event_col, value_col, time_col, dict_port_rel):
    """
    Vectorized extraction of trial features for bandit analysis.
    Returns a DataFrame with trial-wise features.
    """
    n_trials = len(trial_idx)
    trial_starts = trial_idx
    trial_ends = np.append(trial_idx[1:], len(event_col))
    ports = useful_data[trial_idx, 1].astype(int)

    # Preallocate arrays
    poke_in_times = np.full(n_trials, np.nan)
    poke_out_times = np.full(n_trials, np.nan)
    first_lick_times = np.full(n_trials, np.nan)
    last_lick_times = np.full(n_trials, np.nan)
    outcomes = np.full(n_trials, np.nan)
    block_nums = np.zeros(n_trials, dtype=int)
    trial_rewprobs = np.full(n_trials, np.nan)

    # Vectorized block assignment
    block_start_idx = np.where(event_col == 61)[0]
    block_end_idx = np.where(event_col == 62)[0]
    block_bins = np.searchsorted(block_end_idx, trial_starts, side='right')
    block_nums[:] = block_bins

    # Precompute poke event indices for all possible poke events
    poke_event_indices = {poke: np.flatnonzero(event_col == poke) for poke in set(sum(dict_port_rel.values(), []))}

    # Vectorized poke in time (last poke before trial start)
    for port_num, poke_codes in dict_port_rel.items():
        trial_mask = (ports == port_num)
        trial_starts_port = trial_starts[trial_mask]
        if trial_starts_port.size == 0:
            continue
        poke_inds = np.concatenate([poke_event_indices[code] for code in poke_codes if code in poke_event_indices])
        if poke_inds.size == 0:
            continue
        poke_inds.sort()
        last_poke_idx = np.searchsorted(poke_inds, trial_starts_port, side='right') - 1
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
        valid = (insert_pos < poke_inds.size)
        poke_out_times_for_port = np.full(trial_mask.sum(), np.nan)
        poke_out_times_for_port[valid] = time_col[poke_inds[insert_pos[valid]]]
        poke_out_times[trial_mask] = poke_out_times_for_port

    # Vectorized lick detection (use marker 25, fallback to 24, then 22)
    lick_marker = None
    if np.any(event_col == 25):
        lick_marker = 25
    elif np.any(event_col == 24):
        lick_marker = 24
    else:
        lick_marker = 22
    lick_mask = (event_col == lick_marker)
    lick_times = time_col[lick_mask]
    lick_indices = np.flatnonzero(lick_mask)
    first_lick_idx = np.searchsorted(lick_indices, trial_starts, side='left')
    last_lick_idx = np.searchsorted(lick_indices, trial_ends, side='right') - 1
    valid_first = (first_lick_idx < lick_indices.size)
    first_lick_times[valid_first] = lick_times[first_lick_idx[valid_first]]
    valid_last = (last_lick_idx >= 0) & (last_lick_idx < lick_indices.size)
    last_lick_times[valid_last] = lick_times[last_lick_idx[valid_last]]

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

    # Compute rewprobfull (reward probability array per block)
    # For each block, extract all rewardProbMarker events and their values, sort by port, and assign as string to all trials in that block

    # Vectorized rewprobfull assignment
    # Precompute reward probability string for each block
    block_count = max(block_nums) + 1
    block_rewprob_strs = np.empty(block_count, dtype=object)
    for block in range(block_count):
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
    rewprobfull = block_rewprob_strs[block_nums]

    # Assemble DataFrame
    sessdf_vec = pd.DataFrame({
        'trial': np.arange(n_trials),
        'port': ports,
        'poke_in_time': poke_in_times,
        'poke_out_time': poke_out_times,
        'first_lick_time': first_lick_times,
        'last_lick_time': last_lick_times,
        'block_num': block_nums,
        'outcome': outcomes,
        'trial_rewprob': trial_rewprobs,
        'rewprobfull': rewprobfull,
        # Add other columns as needed
    })
    return sessdf_vec

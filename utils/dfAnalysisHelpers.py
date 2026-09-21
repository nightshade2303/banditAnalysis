import numpy as np
import pandas as pd
def get_sessionwise_averages(df, mask = None, metric = None, tasks = None):
    '''Input df and get a dictionary of session-wise averaged metric
    Inputs: df, mask, metric (string), tasks (list)
    Output: Animal-wise session-averaged metric_dict
    '''
    if metric is None:
        assert isinstance(metric, str), "pass a metric column"

    if tasks is not None:
        assert isinstance(tasks, list), "tasks should be list"

        metric_dict = {}
        for task in tasks:
            for ind, (an, group) in enumerate(df[df.task == task].groupby('animal')):
                L = subset_metric_to_numpy(group, metric)
                metric_dict[an, task] = np.nanmean(L, axis = 0)

    else:
        metric_dict = {}
        for ind, (an, group) in enumerate(df.groupby('animal')):
            L = subset_metric_to_numpy(group, metric)
            metric_dict[an] = np.nanmean(L, axis = 0)
            
    return metric_dict

def get_sessionwise_metric(df, mask = None, metric = None, tasks = None):
    '''Input df and get a dictionary of session-wise metric
    Inputs: df, mask, metric (string), tasks (list)
    Output: Animal-wise metric_dict
    '''
    if metric is None:
        assert isinstance(metric, str), "pass a metric column"

    if tasks is not None:
        assert isinstance(tasks, list), "tasks should be list"

        metric_dict = {}
        for task in tasks:
            for _, (an, group) in enumerate(df[df.task == task].groupby('animal')):
                L = subset_metric_to_numpy(group, metric)
                metric_dict[an, task] = L

    else:
        metric_dict = {}
        for _, (an, group) in enumerate(df.groupby('animal')):
            L = subset_metric_to_numpy(group, metric)
            metric_dict[an] = L
    return metric_dict

def subset_metric_to_numpy(group, metric):
    # works only for equal trials in all sessions
    # Select only the columns needed and ensure they're numeric
    temp_df = group[['block_num', metric]].copy()
    
    # Ensure metric column is numeric (convert if needed)
    if not pd.api.types.is_numeric_dtype(temp_df[metric]):
        temp_df[metric] = pd.to_numeric(temp_df[metric], errors='coerce')
    
    g = temp_df.groupby('block_num').cumcount()
    L = (np.array(temp_df.set_index(['block_num', g])
            .unstack(fill_value=np.nan)
            .stack(future_stack=True).groupby(level=0)
            .apply(lambda x: x[metric].values.tolist())
            .tolist()))
    return L

def get_switch_distance_from_ports(ports_sess, remove_empty_rows = False):
    '''
    get switch distance from matrix of ports n_trials x m_session
    '''
    switch_arr = np.abs(np.diff(ports_sess))
    switch_arr = switch_arr[~np.isnan(switch_arr).any(axis=1)]
    switch_arr = switch_arr.astype(float)
    switch_arr[switch_arr == 0] = np.nan
    
    if remove_empty_rows:
    # keep rows without NaNs and align an_ with switch_arr
        mask = ~np.isnan(switch_arr).all(axis=1)
        switch_arr = switch_arr[mask]
    return switch_arr

def calc_dist_metric(tempdf, mask):
    mean_distance = pd.Series([2, 4/3, 4/3, 2], index=[1, 2, 3, 4])

    filtered = tempdf[mask]

    # calculate `d_value` for each group
    d_values = filtered.groupby(['animal', 'block_num'])['disp'].apply(lambda x: x.abs().mean())

    # calculate `d_chance` for each group
    d_chances = filtered.groupby(['animal', 'block_num']).apply(
        lambda group: np.sum(group['port'].value_counts(normalize=True) * mean_distance)
    )

    tempdf['d_value'] = tempdf.set_index(['animal', 'block_num']).index.map(d_values).values
    tempdf['d_chance'] = tempdf.set_index(['animal', 'block_num']).index.map(d_chances).values
    return tempdf

def generate_tm(df, mask = None, col1 = 'port', col2 = 'choice_t1'):
    if mask is not None:
        mat = pd.crosstab(
            df.loc[mask, col1], 
            df.loc[mask, col2], 
            normalize='index'
            )
        return mat
    mat = pd.crosstab(
            df.loc[col1], 
            df.loc[col2], 
            normalize='index'
            )
    return mat

def lzw_compress(
    uncompressed: str,
    return_ratio: bool = False,
    return_alphabet: bool = False,
):
    """Compress a string into LZW integer codes.

    The initial dictionary is built from the characters that appear in
    ``uncompressed`` (in first-seen order), which mirrors the original
    alphabet-based LZW formulation.

    If ``return_ratio`` is True, also return a simple compression ratio
    defined as ``len(uncompressed) / len(compressed_codes)``.

    If ``return_alphabet`` is True, also return the initial alphabet needed
    for exact decompression with ``lzw_decompress(..., alphabet=...)``.
    """
    alphabet = list(dict.fromkeys(uncompressed))
    dictionary = {ch: idx for idx, ch in enumerate(alphabet)}
    dict_size = len(dictionary)
    
    w = ""
    result = []
    
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add the new pattern to the dictionary
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    
    # Output the code for the remaining string
    if w:
        result.append(dictionary[w])

    if return_ratio and return_alphabet:
        ratio = 1.0 if len(uncompressed) == 0 else len(uncompressed) / len(result)
        return result, ratio, alphabet

    if return_ratio:
        ratio = 1.0 if len(uncompressed) == 0 else len(uncompressed) / len(result)
        return result, ratio

    if return_alphabet:
        return result, alphabet

    return result


def lzw_decompress(compressed: list[int], alphabet: list[str] | None = None) -> str:
    """Decompress LZW integer codes back into a string.

    If ``alphabet`` is provided, it should be the initial alphabet returned by
    ``lzw_compress(..., return_alphabet=True)``.
    """
    if alphabet is None:
        # Legacy mode: standard 256 ASCII initialization.
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
    else:
        dict_size = len(alphabet)
        dictionary = {i: ch for i, ch in enumerate(alphabet)}
    
    if not compressed:
        return ""
    
    # Read the first code and output its character
    if compressed[0] not in dictionary:
        raise ValueError(f"Bad first compressed code: {compressed[0]}")
    w = dictionary[compressed[0]]
    result = [w]
    
    for code in compressed[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            # Handle the special LZW edge case (e.g., "cScSc" patterns)
            entry = w + w[0]
        else:
            raise ValueError(f"Bad compressed code: {code}")
            
        result.append(entry)
        
        # Add the new pattern to the dictionary
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        
        w = entry
        
    return "".join(result)


# # --- Example Usage ---
# if __name__ == "__main__":
#     text = "TOBEORNOTTOBEORTOBEORNOT"
    
#     # 1. Compress
#     compressed_data = lzw_compress(text)
#     print("Original Text: ", text)
#     print("Compressed Codes:", compressed_data)
#     print(f"Original Size:   {len(text)} characters")
#     print(f"Compressed Size: {len(compressed_data)} codes")

#     # 2. Decompress
#     decompressed_text = lzw_decompress(compressed_data)
#     print("Decompressed:    ", decompressed_text)
#     print("Matches original:", text == decompressed_text)

"""
Scan every animal's .dat files for the data problems found so far and write CSV summaries.

Checks (per file, then summarised per animal):
  - mixed lick markers across sessions (e.g. 22 in some files, 25 in others) -> the old
    trializer cell picks ONE marker for the merged data and leaves the rest as NaN licks
  - duplicated trial starts (two 81 rows with the same smtime/eptime), and whether the
    which row matches the poked port (the other is the extra row), and whether the reward
    outcome followed the poked port's or the extra row's port probability
  - reward outcomes not following the chosen port's block probability (non-duplicated files)
  - corrupted rows (non-numeric fields, e.g. firmware error messages written into the log)
  - block start/end count mismatch (61 vs 62) -> trializers comparing total 61/62 counts on the
    merged data switch the WHOLE animal to the fallback block assignment
  - reward-probability rows (88) not matching trial starts (81)
  - sessions where most trials have no lick
  - every lick row logged twice
Files with no trials are counted in the notes but are not treated as a problem.

Usage (from the repo root):
    python utils/check_dat_files.py                  # all animals under the roots below
    python utils/check_dat_files.py Xatu Togepi      # only these animals

Outputs (in the working directory):
    dat_check_summary.csv   one row per animal folder; issues == 'ok' if nothing is wrong
    dat_check_files.csv     one row per .dat file with the raw per-file metrics
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.opconNosepokeFunctions import get_files, exclude_files

# folders to scan: (glob pattern for animal folders, subfolder holding the .dat files)
ROOTS = [
    ('L:/4portProb/Box*/*/', ''),
    ('L:/4portProb_ephys/Box*_ephys/*/', 'Behavior/'),
]
SKIP_ANIMALS = set()

NAMES = ['event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww']
SESSION_SM = [12, 13]
LICK_CODES = (28, 25, 24, 22)
CODE2PORT = {21: 1, 20: 2, 17: 3, 19: 3, 37: 4, 18: 4}

# thresholds
HIGH_NAN_LICK = 0.5          # session flagged if more than this fraction of trials have no lick
MIN_TRIALS_FOR_NAN = 100     # ...and it has at least this many trials
MIN_TRIALS_CONTINGENCY = 200  # min trials per probability level for the reward check
CONTINGENCY_TOL = 0.1        # trial-weighted mean |reward rate - probability| before flagging


def read_dat(path):
    raw = pd.read_csv(path, header=None, names=NAMES, dtype=str, on_bad_lines='skip',
                      engine='python')
    num = raw.apply(pd.to_numeric, errors='coerce')
    bad = num.isna().any(axis=1) & raw.notna().all(axis=1)   # present but non-numeric
    bad_in_session = bad & num['sm'].isin(SESSION_SM)
    return num.dropna().reset_index(drop=True), int(bad.sum()), int(bad_in_session.sum())


def check_file(path):
    out = {'file': os.path.basename(path)}
    try:
        df, n_bad, n_bad_sess = read_dat(path)
    except Exception as e:  # unreadable file
        out['read_error'] = repr(e)
        return out, []
    out['n_corrupt_rows'] = n_bad
    out['n_corrupt_rows_in_session'] = n_bad_sess

    df = df[df['sm'].isin(SESSION_SM)].reset_index(drop=True)
    e = df['event'].to_numpy().astype(int)
    v = df['value'].to_numpy().astype(int)
    st = df['smtime'].to_numpy()
    ep = df['eptime'].to_numpy()

    t81 = np.flatnonzero(e == 81)
    out['n81_rows'] = len(t81)
    out['n61'] = int((e == 61).sum())
    out['n62'] = int((e == 62).sum())
    out['n88'] = int((e == 88).sum())
    present = [c for c in LICK_CODES if np.any(e == c)]
    out['lick_codes_present'] = ' '.join(map(str, present))
    marker = next((c for c in LICK_CODES if c in present), None)
    out['lick_marker'] = marker if marker is not None else ''

    # duplicated trial starts: 81 immediately followed by another 81 at the same time
    is_dup = np.zeros(len(e), dtype=bool)
    if len(t81) > 1:
        a, b = t81[:-1], t81[1:]
        pair = (b == a + 1) & (st[a] == st[b]) & (ep[a] == ep[b])
        is_dup[b[pair]] = True
    out['n_dup81'] = int(is_dup.sum())
    out['dup81_values'] = ' '.join(map(str, sorted(set(v[is_dup])))) if is_dup.any() else ''
    real81 = t81[~is_dup[t81]]
    out['n_trials'] = len(real81)
    out['n88_minus_trials'] = out['n88'] - len(real81)

    # for each duplicated pair, the row that matches the port of the last poke-in before it is
    # the poked (real) row; the other is the extra row. Which one comes first differs by animal.
    poke_on_idx = np.flatnonzero(np.isin(e, list(CODE2PORT)) & (v == 1))
    poked_row = {}   # first row of pair -> index of the poked row, or None if unresolved
    for first in np.flatnonzero(is_dup) - 1:
        if is_dup[first]:   # 3+ rows at the same time: only the first pair is resolved
            continue
        second = first + 1
        k = np.searchsorted(poke_on_idx, first) - 1
        port = CODE2PORT[e[poke_on_idx[k]]] if k >= 0 else None
        if v[first] == port and v[second] != port:
            poked_row[first] = first
        elif v[second] == port and v[first] != port:
            poked_row[first] = second
        else:
            poked_row[first] = None
    rows = list(poked_row.items())
    out['n_dup_poked_first'] = sum(1 for f, p in rows if p == f)
    out['n_dup_poked_second'] = sum(1 for f, p in rows if p is not None and p != f)
    out['n_dup_unresolved'] = sum(1 for f, p in rows if p is None)

    # licks logged twice (consecutive identical lick rows)
    if marker is not None:
        lk = np.flatnonzero(e == marker)
        if len(lk) > 1:
            same = (np.diff(lk) == 1) & (st[lk[1:]] == st[lk[:-1]]) & (v[lk[1:]] == v[lk[:-1]])
            out['frac_lick_rows_doubled'] = round(2 * same.sum() / len(lk), 3)

    # fraction of trials with no lick ON between this trial start and the next (all 81 rows,
    # as the trializer does, so duplicated starts show up as missing licks)
    if len(t81) and marker is not None:
        lick = (e == marker) & (v == 1)
        if marker == 22:  # 22 also fires with pokes: drop 22s next to a same-time poke
            poke = np.isin(e, list(CODE2PORT))
            adj = np.zeros(len(e), dtype=bool)
            adj[1:] |= poke[:-1] & (st[1:] == st[:-1])
            adj[:-1] |= poke[1:] & (st[:-1] == st[1:])
            lick &= ~adj
        lick_idx = np.flatnonzero(lick)
        ends = np.append(t81[1:], len(e))
        has = np.searchsorted(lick_idx, ends, 'left') > np.searchsorted(lick_idx, t81, 'right')
        out['frac_trials_no_lick'] = round(1 - has.mean(), 3)

    # reward contingency records per trial: (p_first, p_second, rewarded, is_dup_trial, poked_is_first)
    # p_first / p_second = block probability of the port in the first / second 81 row of a duplicated
    # pair (p_second is NaN for normal trials); unresolved pairs are skipped
    recs = []
    probs = {}
    real_set = set(real81.tolist())
    for i in range(len(e)):
        if e[i] == 61:
            probs = {}
        elif e[i] == 83:
            probs[v[i]] = st[i]
        elif e[i] == 81 and i in real_set:
            dup = i + 1 < len(e) and is_dup[i + 1]
            if dup and poked_row.get(i) is None:
                continue
            j = i + (2 if dup else 1)
            while j < len(e) and e[j] not in (81, 51, 86):
                j += 1
            if j < len(e) and e[j] in (51, 86) and probs:
                if dup:
                    recs.append((probs.get(v[i], np.nan), probs.get(v[i + 1], np.nan),
                                 int(e[j] == 51), True, poked_row[i] == i))
                else:
                    recs.append((probs.get(v[i], np.nan), np.nan, int(e[j] == 51), False, np.nan))
    return out, recs


def contingency(recs, col):
    """Reward rate per probability level; returns (trial-weighted mean |rate - p|, table) or None."""
    r = recs.dropna(subset=[col])
    if r.empty:
        return None
    g = r.groupby(col)['rew'].agg(['mean', 'size'])
    g = g[g['size'] >= MIN_TRIALS_CONTINGENCY]
    if len(g) < 2:
        return None
    dev = np.average((g['mean'] - g.index.to_numpy() / 100).abs(), weights=g['size'])
    table = ', '.join(f'{int(p)}%->{m:.0%}' for p, m in g['mean'].items())
    return dev, table


def summarise(animal, folder, file_rows, recs):
    fr = pd.DataFrame(file_rows)
    issues, notes = [], []
    n_files = len(fr)
    if n_files == 0:
        return dict(animal=animal, folder=folder, n_files=0, issues='no .dat files found', notes='')

    for col in ['n_trials', 'n_dup81', 'n61', 'n62', 'n88_minus_trials', 'n_corrupt_rows',
                'n_corrupt_rows_in_session', 'frac_trials_no_lick', 'frac_lick_rows_doubled',
                'lick_marker', 'dup81_values', 'n_dup_poked_first', 'n_dup_poked_second',
                'n_dup_unresolved']:
        if col not in fr:
            fr[col] = np.nan
    fr['n_trials'] = fr['n_trials'].fillna(0)
    if 'read_error' in fr and fr['read_error'].notna().any():
        bad = fr[fr['read_error'].notna()]
        issues.append(f"{len(bad)} unreadable files (e.g. {bad['file'].iloc[0]})")

    wt = fr[fr['n_trials'] > 0]
    notes.append(f"{n_files - len(wt)} of {n_files} files have no trials")

    # mixed lick markers
    markers = pd.to_numeric(wt['lick_marker'], errors='coerce').dropna().astype(int)
    counts = markers.value_counts().sort_index()
    if len(counts) > 1:
        issues.append('mixed lick markers across sessions (' +
                      ', '.join(f'{m} in {n} files' for m, n in counts.items()) +
                      ') - a trializer that picks one lick marker for all merged sessions '
                      '(e.g. sessdf_vec in rodentBanditAnalysis.ipynb) leaves the rest as NaN licks')
    no_marker = wt[pd.to_numeric(wt['lick_marker'], errors='coerce').isna()]
    if len(no_marker):
        issues.append(f"{len(no_marker)} files with trials but no lick events")

    # duplicated trial starts
    dup = wt[wt['n_dup81'] > 0]
    if len(dup):
        vals = sorted(set(' '.join(dup['dup81_values'].astype(str)).split()))
        n_first = int(dup['n_dup_poked_first'].sum())
        n_second = int(dup['n_dup_poked_second'].sum())
        n_unres = int(dup['n_dup_unresolved'].sum())
        n_pairs = n_first + n_second + n_unres
        msg = (f"duplicated trial starts (81) in {len(dup)} files "
               f"({dup['file'].min()} .. {dup['file'].max()}; 81 values seen in the duplicates: {' '.join(vals)}); "
               f"row matching the poked port is FIRST in {n_first}/{n_pairs} pairs, "
               f"SECOND in {n_second}/{n_pairs}, unresolved {n_unres}/{n_pairs}")
        if n_second > n_first:
            msg += (" - trializers that keep the first 81 record the extra row's port "
                    "instead of the poked port")
        # which row's port probability did the reward follow? (checked over all duplicated trials)
        dr = recs[recs['dup']]
        by_first = contingency(dr, 'p_first')
        by_second = contingency(dr, 'p_second')
        if by_first and by_second:
            reward_row = 'second' if by_second[0] < by_first[0] else 'first'
            table = by_second[1] if reward_row == 'second' else by_first[1]
            # trials rewarded by the wrong port = pairs where the reward row is not the poked row
            n_wrong = n_first if reward_row == 'second' else n_second
            n_resolved = n_first + n_second
            msg += (f"; reward followed the {reward_row.upper()} row's port probability ({table}), so "
                    f"{n_wrong}/{n_resolved} ({n_wrong / n_resolved:.0%}) duplicated trials were rewarded "
                    f"using the wrong port's probability")
        else:
            msg += '; too few duplicated trials to tell which row the reward followed'
        issues.append(msg)

    # reward contingency on normal trials
    ok = contingency(recs[~recs['dup']], 'p_first')
    if ok and ok[0] > CONTINGENCY_TOL:
        issues.append(f"reward rate does not match the poked port's probability ({ok[1]})")

    # corrupted rows
    cor = fr[fr['n_corrupt_rows'] > 0]
    if len(cor):
        n_sess = int(fr['n_corrupt_rows_in_session'].sum())
        if n_sess:
            issues.append(f"{n_sess} corrupted (non-numeric) rows inside task state machines "
                          f"in {int((fr['n_corrupt_rows_in_session'] > 0).sum())} files - these rows are dropped")
        else:
            notes.append(f"{len(cor)} files contain corrupted rows outside the task state "
                         f"machines (filtered out; only causes DtypeWarning)")

    # block markers
    mm = wt[wt['n61'] != wt['n62']]
    if len(mm):
        issues.append(f"block start/end count mismatch (61 vs 62) in {len(mm)} files "
                      f"(e.g. {mm['file'].iloc[0]}) - trializers comparing total 61/62 counts "
                      f"on the merged data then use the fallback block assignment for ALL sessions")

    # reward-probability rows
    m88 = wt[wt['n88_minus_trials'] != 0]
    if len(m88):
        worst = int(m88['n88_minus_trials'].abs().max())
        issues.append(f"88 (trial reward prob) count != trial count in {len(m88)} files "
                      f"(max difference {worst})")

    # sessions with mostly no licks
    hn = wt[(wt['frac_trials_no_lick'] > HIGH_NAN_LICK) & (wt['n_trials'] >= MIN_TRIALS_FOR_NAN)
            & (wt['n_dup81'] == 0)]
    if len(hn):
        issues.append(f"{len(hn)} sessions (>= {MIN_TRIALS_FOR_NAN} trials) with no lick on "
                      f"> {HIGH_NAN_LICK:.0%} of trials (e.g. {hn['file'].iloc[0]})")

    # doubled lick rows
    dl = wt[wt['frac_lick_rows_doubled'] > 0.9]
    if len(dl):
        notes.append(f"every lick row logged twice in {len(dl)} files "
                     f"(harmless for lick times, doubles lick counts)")

    return dict(
        animal=animal, folder=folder, n_files=n_files, n_files_with_trials=len(wt),
        n_trials=int(wt['n_trials'].sum()),
        first_file=wt['file'].min() if len(wt) else '', last_file=wt['file'].max() if len(wt) else '',
        lick_markers=' '.join(f'{m}:{n}' for m, n in counts.items()),
        issues=' | '.join(issues) if issues else 'ok',
        notes=' | '.join(notes),
    )


def animal_folders(only=None):
    for pattern, sub in ROOTS:
        for fol in sorted(glob.glob(pattern)):
            animal = os.path.basename(os.path.normpath(fol))
            if animal in SKIP_ANIMALS or (only and animal not in only):
                continue
            path = os.path.join(fol, sub).replace('\\', '/')
            if os.path.isdir(path):
                yield animal, path


def main():
    only = set(sys.argv[1:]) or None
    summary, all_files = [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for animal, folder in animal_folders(only):
            files = sorted(exclude_files(get_files(folder)))
            results = list(pool.map(check_file, [folder + f for f in files], chunksize=4))
            file_rows = [r for r, _ in results]
            recs = pd.DataFrame([x for _, rs in results for x in rs],
                                columns=['p_first', 'p_second', 'rew', 'dup', 'poked_is_first'])
            recs['dup'] = recs['dup'].astype(bool)
            for r in file_rows:
                r.update(animal=animal, folder=folder)
            all_files += file_rows
            row = summarise(animal, folder, file_rows, recs)
            summary.append(row)
            print(f"{animal:15s} {folder:45s} {row['issues'][:120]}", flush=True)

            # write as we go so a crash/interrupt keeps what has been done
            pd.DataFrame(summary).to_csv('dat_check_summary.csv', index=False)
            files_df = pd.DataFrame(all_files)
            fcols = ['animal', 'folder', 'file'] + [c for c in files_df.columns
                                                     if c not in ('animal', 'folder', 'file')]
            files_df[fcols].to_csv('dat_check_files.csv', index=False)


if __name__ == '__main__':
    main()

"""Loading and cleaning helpers for the trial tables (sessdf, hsbdf).

Used by the paper figures (through utils/load_data.py): read_df, subset_trials, add_block_groups, add_day_task,
add_shift_info, remove_sessions_with_wrong_rewards, remove_sessions_with_duplicate_trials. The other functions
(expert_sessions, remove_duplicates, describe_dataset) are kept for the analysis notebooks.

add_block_groups, add_day_task and add_shift_info add columns to the table they are given (in place) and return
it, as the notebooks expect.
"""
import pandas as pd

SESSION_BIN = 50          # sess_bin: sessions are numbered in bins of this many (used by expert_sessions)
SESSION_GAP_S = 40 * 60   # a gap longer than this (seconds) between blocks or trials starts a new session
# the four structured (correlated) reward profiles, % at ports 1-4 as written in rewprobfull
STRUCTURED_PROFILES = ['[80 46 15 10]', '[46 80 46 15]', '[15 46 80 46]', '[10 15 46 80]']
# reward profiles of an older assignment of 'structured' sessions; kept as the 'assigned' column for the notebooks
ASSIGNED_STR_PROFILES = ['[10 15 46 80]', '[10 15 80 46]', '[10 46 80 15]', '[10 80 46 15]', '[15 10 46 80]',
                         '[15 10 80 46]', '[15 46 46 80]', '[15 46 80 10]', '[15 46 80 46]', '[15 80 46 10]',
                         '[15 80 46 46]', '[46 46 80 15]', '[46 80 10 15]', '[46 80 15 10]', '[46 80 46 15]',
                         '[80 46 10 15]', '[80 46 15 10]', '[80 46 46 15]']
# malformed reward profiles (test or broken blocks), dropped on reading
BAD_PROFILES = ['[ 20  20  20 100]', '[0 0 0 0]', '[0]', '[0 0]', '[1000   80]', '[30]', '[40]', '[70]']


def read_df(filepath, task_type=None, animals=None):
    """Read a sessdf csv. task_type: list of tasks to keep (default: 'unstr' only); animals: rats to keep (the
    file is then read in chunks, as it is large). Blocks with a malformed reward profile are dropped. Adds:
    sess_bin (session // SESSION_BIN + 1), week_task (weeks since the rat's first trial of the task), env
    ('str' if the block's profile is one of STRUCTURED_PROFILES, else 'unstr') and assigned (older
    assignment, see ASSIGNED_STR_PROFILES)."""
    if animals is None:
        df = pd.read_csv(filepath)
    else:
        chunks = pd.read_csv(filepath, chunksize=2_000_000)
        df = pd.concat([chunk[chunk.animal.isin(animals)] for chunk in chunks])
    df = df.drop(columns='Unnamed: 0')

    df['sess_bin'] = df.groupby(['animal', 'task'])['session'].transform(
        lambda s: pd.cut(s, bins=range(0, s.max() + SESSION_BIN, SESSION_BIN), labels=False, right=False) + 1)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['week_task'] = df.groupby(['animal', 'task'])['datetime'].transform(lambda t: (t - t.min()).dt.days // 7 + 1)
    df['assigned'] = df.rewprobfull.isin(ASSIGNED_STR_PROFILES).map({True: 'str', False: 'unstr'})
    df['env'] = df.rewprobfull.isin(STRUCTURED_PROFILES).map({True: 'str', False: 'unstr'})
    df = df[~df.rewprobfull.isin(BAD_PROFILES)]

    if task_type is None:
        return df[df.task == 'unstr']
    assert isinstance(task_type, list), 'task_type should be a list'
    return df[df.task.isin(task_type)]


def subset_trials(df, trialsinsess, head_trials=None, tail_trials=None):
    """Blocks with at least `trialsinsess` trials (despite the name, trials in the block), optionally cut to
    their first head_trials and/or last tail_trials trials."""
    blocks = df.groupby(['animal', 'block_num'])
    subset = df[blocks.reward.transform('size') >= trialsinsess]
    for n, cut in [(head_trials, 'head'), (tail_trials, 'tail')]:
        if n is not None:
            assert isinstance(n, int) and n <= trialsinsess, f'{cut}_trials must be an int <= trialsinsess'
            subset = getattr(subset.groupby(['animal', 'block_num']), cut)(n)
    return subset


def expert_sessions(df, sess_bin = 4):
    return df[df.sess_bin>=4]

def remove_duplicates(df, trial_limit = 5):
    # remove all sessions with duplicates in them
    # if there are <5 in a session, remove first line. else remove that session
    l = []
    for animal, group in df.groupby('animal'):
        mask = group.duplicated(subset = ['animal', 'session', 'trialstart', 'eptime'], keep = 'last')
        remove_sess_num = group[mask].groupby('session').trial.nunique()[group[mask].groupby('session').trial.nunique()>5].index
        l.append(df[(df.animal==animal) & (~df.session.isin(remove_sess_num))])
    cleandf = pd.concat(l)
    cleandf = cleandf[~cleandf.duplicated(subset = ['animal', 'session', 'trialstart', 'eptime'], keep = 'last')]
    return cleandf

def remove_sessions_with_duplicate_trials(df, animal_col='animal', session_col='session', max_duplicates=10, verbose=True):
    # drop sessions with more than max_duplicates duplicated trial rows: rows of the same session with the
    # same poke-in (or first-lick) time, block and eptime. Tables without poke times (older trializer,
    # e.g. hsb_data) use trialstart and trialend instead
    time_cols = ['poke_in_time', 'first_lick_time'] if 'poke_in_time' in df.columns else ['trialstart', 'trialend']
    keys = [animal_col, session_col]
    n_dup = None
    for col in time_cols:
        valid = df[df[col].notna()]
        dup = valid.duplicated(subset = keys + [col, 'block_num', 'eptime'], keep = False)
        counts = dup.groupby([valid[k] for k in keys]).sum()
        n_dup = counts if n_dup is None else n_dup.add(counts, fill_value = 0)
    to_remove = n_dup[n_dup > max_duplicates]
    if verbose:
        for (animal, session), n in to_remove.items():
            print(f'removed {animal, session, int(n)}')
    keep = ~pd.MultiIndex.from_frame(df[keys]).isin(to_remove.index)
    return df[keep].copy()

# old name, still used by the notebooks
remove_sessions_with_few_duplicate_trials = remove_sessions_with_duplicate_trials

def remove_sessions_with_wrong_rewards(df, animal_col='animal', session_col='session'):
    # drop whole sessions with any trial rewarded from another port's probability (rewarded_by_wrong_port,
    # see docs/trializer_v4_synced.md): the rat experienced those rewards, so the rest of the session is affected too
    keys = [animal_col, session_col]
    bad = pd.MultiIndex.from_frame(df.loc[df.rewarded_by_wrong_port == True, keys])
    keep = ~pd.MultiIndex.from_frame(df[keys]).isin(bad)
    return df[keep].copy()

def describe_dataset(df, mask=None):
    if mask is not None:
        return df[mask].groupby(['animal', 'session'], as_index = False)['trial'].count().describe()
    return df.groupby(['animal', 'session'], as_index = False)['trial'].count().describe()

def add_block_groups(df):
    """Split each rat's trials into sessions and number the blocks within each session.

    A new session starts at a block that begins more than SESSION_GAP_S after the previous block began, and at
    any gap of more than SESSION_GAP_S between consecutive trials, even inside a block (a block can carry on
    after the rat stopped, e.g. overnight). Adds sess_block (session number, counting from 0 per rat and task)
    and block_group (position of the block in its session, 1 = first). df must have a unique index."""
    def gap_s(times):  # seconds since the previous time; total_seconds keeps whole days, unlike .dt.seconds
        return times.astype('datetime64[ns]').diff().dt.total_seconds()

    block_start = df.groupby(['animal', 'task', 'block_num']).datetime.head(1)
    new_block_session = gap_s(block_start) > SESSION_GAP_S
    new_trial_session = df.groupby(['animal', 'task']).datetime.transform(gap_s) > SESSION_GAP_S

    df['sess_block'] = 0
    df.loc[new_block_session[new_block_session].index, 'sess_block'] = 1
    df.loc[new_trial_session, 'sess_block'] = 1
    df['sess_block'] = df.groupby(['animal', 'task']).sess_block.cumsum()
    df['block_group'] = (df.groupby(['animal', 'task', 'sess_block'])['block_num']
                         .transform(lambda blocks: pd.factorize(blocks)[0] + 1))
    return df

def _bias_dates(cfg):
    """{animal: sorted array of bias-correction start times} from a rules json.
    Each entry is a 'YYYY-MM-DD' string or {"from": ..., "to": ...} (inclusive); each day's
    bias correction starts at "bias_start_time" (default 19:00) on that date."""
    start = pd.Timedelta(cfg.get('bias_start_time', '19:00') + ':00')
    out = {}
    for animal, entries in cfg.get('bias_correction', {}).items():
        dates = []
        for e in entries:
            if isinstance(e, dict):
                dates.extend(pd.date_range(e['from'], e['to'], freq='D'))
            else:
                dates.append(pd.Timestamp(e))
        out[animal] = (pd.DatetimeIndex(dates).normalize().unique().sort_values() + start).values
    return out


def add_day_task(df, rules_path=None):
    """Day of training, starting at 1.

    rules_path=None: 24-hour periods since the animal's first trial (original behaviour).
    rules_path given (e.g. hsb_task_data.json): calendar days since the start of the phase,
    except that a gap between sessions containing bias-correction days counts as 1 day, so the
    first session after bias correction is the day after the last session before it. The count
    restarts on entering any task in "reset_day_count_on" (e.g. ph_implant), so call this after
    apply_task_rules_from_json. week_task is recomputed from the adjusted day_task.
    """
    if rules_path is None:
        df['day_task'] = df.groupby('animal')['datetime'].transform(lambda x: ((x - x.min()).dt.days) + 1)
        return df

    import json
    import numpy as np
    with open(rules_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    bias = _bias_dates(cfg)
    reset_tasks = set(cfg.get('reset_day_count_on', []))

    df['datetime'] = pd.to_datetime(df['datetime'])
    date = df['datetime'].dt.normalize()
    order = df.sort_values(['animal', 'datetime']).index

    # phase: increments each time the animal enters a reset task (e.g. ph_implant)
    task = df.loc[order, 'task'].astype(str)
    entered = task.ne(task.groupby(df.loc[order, 'animal'].astype(str)).shift()) & task.isin(reset_tasks)
    phase = entered.astype(int).groupby(df.loc[order, 'animal'].astype(str)).cumsum().reindex(df.index)

    day = pd.Series(np.nan, index=df.index)
    for (animal, _), idx in df.groupby([df['animal'].astype(str), phase], observed=True).groups.items():
        d = date.loc[idx]
        u = np.sort(d.unique())  # session dates of this phase
        step = np.diff(u).astype('timedelta64[D]').astype(int)  # calendar days between sessions
        b = bias.get(animal)
        if b is not None and len(b):
            # a gap containing a bias-correction start (e.g. 19:00 on a bias date) collapses to 1:
            # the first session after bias correction is the day after the last session before it
            t = df.loc[idx, 'datetime'].groupby(d.values)
            last_t, first_t = t.max().reindex(u).values, t.min().reindex(u).values
            n_bias = np.searchsorted(b, first_t[1:], 'right') - np.searchsorted(b, last_t[:-1], 'right')
            step = np.where(n_bias > 0, 1, step)
        day_of = dict(zip(u, np.concatenate([[1], 1 + np.cumsum(step)])))
        day.loc[idx] = d.map(day_of).values
    df['day_task'] = day.astype(int)
    df['week_task'] = (df['day_task'] - 1) // 7 + 1
    return df

def add_shift_info(df):
    """Next choices and switches, within each session. Adds:
    choice_t1, choice_t2: port chosen 1 and 2 trials later (a recorded port 0 counts as the current port);
    disp: choice_t1 - port (signed switch distance, 0 on a stay);
    shift_t0: 1 if choice_t1 is a different port (switch), 0 if the same (stay), NaN on the session's last trial;
    shift_t1: the same for choice_t2."""
    for col, lag in [('choice_t1', 1), ('choice_t2', 2)]:
        df[col] = df.groupby(['animal', 'session']).port.shift(-lag)
        no_port = df[col] == 0
        df.loc[no_port, col] = df.loc[no_port, 'port']
    df['disp'] = df['choice_t1'] - df['port']
    df['shift_t0'] = (df['choice_t1'] != df['port']).astype(float).where(df['choice_t1'].notna())
    df['shift_t1'] = (df['choice_t2'] != df['port']).astype(float).where(df['choice_t2'].notna())
    return df


# utility functions
import pandas as pd

# read cleaned dataframe with no duplicates+unstructured data only
def read_df(filepath, task_type = None, animals = None):
    if animals is None:
        df = pd.read_csv(filepath)
    else:
        # the full file is large, so read it in chunks and keep only these animals
        chunks = pd.read_csv(filepath, chunksize = 2_000_000)
        df = pd.concat([chunk[chunk.animal.isin(animals)] for chunk in chunks])
    df.drop(columns = 'Unnamed: 0', inplace = True)

    # add sess bin info and week info
    bin_size = 50
    df['sess_bin'] = df.groupby(['animal', 'task'])['session'].transform(
        lambda x: pd.cut(
            x, 
            bins=range(0, x.max() + bin_size, bin_size), 
            labels=False, 
            right=False
            ) + 1)
    # # ## week relative to each animal-task first date
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['week_task'] = df.groupby(['animal', 'task'])['datetime'].transform(
        lambda x: ((x - x.min()).dt.days // 7) + 1
    )
    l_str_sess = [
        '[10 15 46 80]',
        '[10 15 80 46]',
        '[10 46 80 15]',
        '[10 80 46 15]',
        '[15 10 46 80]',
        '[15 10 80 46]', 
        '[15 46 46 80]', 
        '[15 46 80 10]',
        '[15 46 80 46]',
        '[15 80 46 10]',
        '[15 80 46 46]',
        '[46 46 80 15]',
        '[46 80 10 15]',
        '[46 80 15 10]',
        '[46 80 46 15]',
        '[80 46 10 15]',
        '[80 46 15 10]',
        '[80 46 46 15]'
    ]
    df['assigned'] = df.rewprobfull.isin(l_str_sess).map({True: 'str', False: 'unstr'})

    struc = ['[80 46 15 10]', '[46 80 46 15]', '[15 46 80 46]', '[10 15 46 80]']
    df['env'] = df.rewprobfull.isin(struc).map({True: 'str', False: 'unstr'})

    exclude = ['[ 20  20  20 100]', '[0 0 0 0]', '[0]', '[0 0]',
        '[1000   80]', '[30]', '[40]', '[70]']
    df = df[~df.rewprobfull.isin(exclude)]

    if task_type is not None:
        assert isinstance(task_type, list), "task_type should be a list"
        df = df[df.task.isin(task_type)]
    else:
        df = df[df.task == 'unstr']
        assert df.task.unique().shape[0]==1
    return df

# for given df, reduce it to certain number of trials for all the given sessions
def subset_trials(df, trialsinsess, head_trials = None, tail_trials = None):
    subset = df.groupby(['animal','block_num']).filter(lambda x: x.reward.size >= trialsinsess)
    if head_trials is not None:
        assert head_trials<=trialsinsess
        assert isinstance(head_trials, int), "num_trials should be int"
        subset = subset.groupby(['animal', 'block_num']).head(head_trials)

    if tail_trials is not None:
        assert tail_trials<=trialsinsess
        assert isinstance(tail_trials, int), "num_trials should be int"
        subset = subset.groupby(['animal', 'block_num']).tail(tail_trials)
    
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
    # make a column called sess_block
    # if session changes within 40min,  number +1
    # first reset index or everything will go crazy
    # df = df.reset_index()
    df['sess_block'] = 0
    # total_seconds, not .dt.seconds: .dt.seconds drops whole days, so a block starting the
    # next day within 40 min of the previous block's clock time was not a new session
    thing = df.groupby(['animal', 'task', 'block_num']).datetime.head(1).astype('datetime64[ns]').diff().dt.total_seconds()>2400
    df.loc[thing[thing == True].index, 'sess_block'] = 1
    df['sess_block'] = df.groupby(['animal', 'task']).sess_block.cumsum()
    # Create a group identifier that increases every time sess_block == 1
    df['block_group'] = df.groupby(['animal', 'task', 'block_num']).ngroup()

    df['block_group'] = (
        df.groupby(['animal', 'task', 'sess_block'])['block_num']
        .transform(lambda x: pd.factorize(x)[0] + 1)
    )
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
    df['choice_t1'] = df.groupby(['animal','session']).port.shift(-1)
    df['choice_t2'] = df.groupby(['animal','session']).port.shift(-2)
    df.loc[(df.choice_t1 == 0), 'choice_t1'] = df.loc[(df.choice_t1 == 0), 'port']
    df.loc[(df.choice_t2 == 0), 'choice_t2'] = df.loc[(df.choice_t2 == 0), 'port']
    df['disp'] = df['choice_t1']-df['port']

    # 1 = switch, 0 = stay, NaN = no next trial in the session
    df['shift_t0'] = (df['choice_t1'] != df['port']).astype(float).where(df['choice_t1'].notna())
    df['shift_t1'] = (df['choice_t2'] != df['port']).astype(float).where(df['choice_t2'].notna())
    return df


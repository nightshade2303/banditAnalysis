# utility functions
import pandas as pd

# read cleaned dataframe with no duplicates+unstructured data only
def read_df(filepath, task_type = None):
    df = pd.read_csv(filepath)
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
    df['assigned'] = df.rewprobfull.isin(l_str_sess)
    df.assigned.replace({True: 'str', False: 'unstr'}, inplace = True)

    struc = ['[80 46 15 10]', '[46 80 46 15]', '[15 46 80 46]', '[10 15 46 80]']
    df['env'] = df.rewprobfull.isin(struc)
    df.env.replace({True: 'str', False: 'unstr'}, inplace = True)

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

def remove_sessions_with_few_duplicate_trials(df, animal_col='animal', session_col='session', max_duplicates=10):
    # Find sessions (per animal) with duplicated trials based on 'trialstart' and 'eptime'
    if 'poke_in_time' in df.columns:
        trial_cols = ['poke_in_time', 'first_lick_time', 'block_num']
    else:
        trial_cols = ['trialstart', 'trialend', 'block_num']
    to_remove = set()
    for (animal, session), group in df.groupby([animal_col, session_col]):
        if 'poke_in_time' in df.columns:
            poke_valid = group[group['poke_in_time'].notna()]
            lick_valid = group[group['first_lick_time'].notna()]

            duplicate_count = (
                poke_valid.duplicated(
                    subset=['poke_in_time', 'block_num'],
                    keep=False
                ).sum()
                + lick_valid.duplicated(
                    subset=['first_lick_time', 'block_num'],
                    keep=False
                ).sum()
            )
        else:
            valid_start = group[group['trialstart'].notna()]
            valid_end = group[group['trialend'].notna()]

            duplicate_count = (
                valid_start.duplicated(
                    subset=['trialstart', 'block_num'],
                    keep=False
                ).sum()
                + valid_end.duplicated(
                    subset=['trialend', 'block_num'],
                    keep=False
                ).sum()
            )
    if duplicate_count > max_duplicates:
        to_remove.add((animal, session))
    # Build mask for rows to keep
    mask = ~(df[[animal_col, session_col]].apply(tuple, axis=1).isin(to_remove))
    return df[mask]

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
    thing = df.groupby(['animal', 'task', 'block_num']).datetime.head(1).astype('datetime64[ns]').diff().dt.seconds>2400
    df.loc[thing[thing == True].index, 'sess_block'] = 1
    df['sess_block'] = df.groupby(['animal', 'task']).sess_block.cumsum()
    # Create a group identifier that increases every time sess_block == 1
    df['block_group'] = df.groupby(['animal', 'task', 'block_num']).ngroup()

    df['block_group'] = (
        df.groupby(['animal', 'task', 'sess_block'])['block_num']
        .transform(lambda x: pd.factorize(x)[0] + 1)
    )
    return df

def add_shift_info(df):
    df['choice_t1'] = df.groupby(['animal','block_num']).port.shift(-1)
    df['choice_t2'] = df.groupby(['animal','block_num']).port.shift(-2)
    df.loc[(df.choice_t1 == 0), 'choice_t1'] = df.loc[(df.choice_t1 == 0), 'port']
    df.loc[(df.choice_t2 == 0), 'choice_t2'] = df.loc[(df.choice_t2 == 0), 'port']
    df['disp'] = df['choice_t1']-df['port']

    df['shift_t0'] = (df['choice_t1']==df['port']).replace({True: 0, False: 1})
    df['shift_t1'] = (df['choice_t2']==df['port']).replace({True: 0, False: 1})
    return df


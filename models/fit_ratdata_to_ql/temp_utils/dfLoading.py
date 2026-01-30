# utility functions
import pandas as pd

# read cleaned dataframe with no duplicates+unstructured data only
def read_df(filepath, task_type = None):
    df = pd.read_csv(filepath)
    df.drop(columns = 'Unnamed: 0', inplace = True)
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
    subset = df.groupby(['animal','session']).filter(lambda x: x.reward.size >= trialsinsess)
    if head_trials is not None:
        assert head_trials<=trialsinsess
        assert isinstance(head_trials, int), "num_trials should be int"
        subset = subset.groupby(['animal', 'session']).head(head_trials)

    if tail_trials is not None:
        assert tail_trials<=trialsinsess
        assert isinstance(tail_trials, int), "num_trials should be int"
        subset = subset.groupby(['animal', 'session']).head(head_trials)
    
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


def describe_dataset(df, mask=None):
    if mask is not None:
        return df[mask].groupby(['animal', 'session'], as_index = False)['trial'].count().describe()
    return df.groupby(['animal', 'session'], as_index = False)['trial'].count().describe()

def add_block_groups(df):
    # make a column called sess_block
    # if session changes within 40min,  number +1
    df['sess_block'] = 0
    thing = df.groupby(['animal', 'task', 'session']).datetime.head(1).astype('datetime64[ns]').diff().dt.seconds>2400
    df.loc[thing[thing == True].index, 'sess_block'] = 1
    df['sess_block'] = df.groupby(['animal', 'task']).sess_block.cumsum()
    # Create a group identifier that increases every time sess_block == 1
    df['block_group'] = df.groupby(['animal', 'task', 'session']).ngroup()

    df['block_group'] = (
        df.groupby(['animal', 'task', 'sess_block'])['session']
        .transform(lambda x: pd.factorize(x)[0] + 1)
    )
    return df

def add_shift_info(df):
    df['choice_t1'] = df.groupby(['animal','session']).port.shift(-1)
    df['choice_t2'] = df.groupby(['animal','session']).port.shift(-2)
    df.loc[(df.choice_t1 == 0), 'choice_t1'] = df.loc[(df.choice_t1 == 0), 'port']
    df.loc[(df.choice_t2 == 0), 'choice_t2'] = df.loc[(df.choice_t2 == 0), 'port']
    df['disp'] = df['choice_t1']-df['port']

    df['shift_t0'] = (df['choice_t1']==df['port']).replace({True: 0, False: 1})
    df['shift_t1'] = (df['choice_t2']==df['port']).replace({True: 0, False: 1})
    return df
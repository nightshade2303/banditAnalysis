"""Load the two cohorts as trial tables with the same columns (see docs/data_columns.md)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from utils.cohorts import COHORTS, GROUPS, UNBLOCKED
from utils.config import EXPERT_AFTER_DAY, HSB_DIR, MAX_DUPLICATES, MIN_BLOCKS, N_TRIALS, RULES_PATH, SESSDF_PATH
from utils.dfLoading import (add_block_groups, add_day_task, add_shift_info, read_df,
                             remove_sessions_with_duplicate_trials, remove_sessions_with_wrong_rewards)
from utils.trializer_v4 import apply_task_rules_from_json

COLUMNS = ['animal', 'task', 'env', 'day_task', 'week_task', 'datetime', 'session', 'block_num', 'block_group',
           'trial', 'port', 'reward', 'rewprob', 'rewprobfull', 'choice_t1', 'disp', 'shift_t0']


def load_sessdf(path=SESSDF_PATH, animals=COHORTS['unstr']):
    """Uncorrelated rats: their 'unstr' trials in sessdf (notebook cell 1).

    Sessions come from add_block_groups, as for hsb_data. Sessions with any wrongly rewarded trial or
    with more than MAX_DUPLICATES duplicated trial rows are dropped."""
    df = read_df(path, task_type=['unstr'], animals=animals).copy()
    df = add_block_groups(df).rename(columns={'trial_rewprob': 'rewprob'})
    df['session'] = df['sess_block']
    df = add_day_task(df)
    df = remove_sessions_with_wrong_rewards(df)
    df = remove_sessions_with_duplicate_trials(df, max_duplicates=MAX_DUPLICATES, verbose=False)
    return add_shift_info(df)[COLUMNS]


def load_hsb(hsb_dir=HSB_DIR, animals=COHORTS['str']):
    """Correlated rats: one csv per rat (notebook cell 3). Post-implant sessions get task 'ph_implant'.

    Sessions with more than MAX_DUPLICATES duplicated trial rows are dropped."""
    df = pd.concat([pd.read_csv(Path(hsb_dir) / f'{an}.csv').assign(animal=an) for an in animals],
                   ignore_index=True)
    # the csv's session column numbers blocks, and its task column says whether a block is structured
    df = df.drop(columns='sess_block').rename(columns={'session': 'block_num', 'task': 'env'})
    df['task'] = 'str'
    df['trial'] = np.arange(len(df))
    df['datetime'] = pd.to_datetime(df.eptime, unit='s')
    df = add_block_groups(df).rename(columns={'sess_block': 'session'})
    df['week_task'] = df.groupby(['animal', 'task'])['datetime'].transform(
        lambda x: ((x - x.min()).dt.days // 7) + 1)
    df = add_day_task(df)
    df = apply_task_rules_from_json(df, rules_path=RULES_PATH)
    df = remove_sessions_with_duplicate_trials(df, max_duplicates=MAX_DUPLICATES, verbose=False)
    return add_shift_info(df)[COLUMNS]


@lru_cache(maxsize=None)
def _cohort_tables():
    return {'unstr': load_sessdf(), 'str': load_hsb()}


def select_trials(df, task, after_day=EXPERT_AFTER_DAY, block_group=None):
    """Trials of one task after `after_day` days of training; block_group=1 keeps each session's first block."""
    mask = (df.task == task) & (df.day_task > after_day)
    if block_group is not None:
        mask &= df.block_group == block_group
    return df[mask].copy()


def rats_with_enough_blocks(df, min_blocks=MIN_BLOCKS, n_trials=N_TRIALS):
    """Rats with more than min_blocks blocks of at least n_trials trials in df."""
    block_len = df.groupby(['animal', 'block_num']).size()
    n_long = (block_len >= n_trials).groupby('animal').sum()
    return n_long.index[n_long > min_blocks]


def _select(df, task, after_day, block_group, min_blocks):
    """select_trials, keeping rats whose data (all blocks, before block_group is applied) have more than
    min_blocks blocks of at least N_TRIALS trials; min_blocks=None keeps all rats."""
    trials = select_trials(df, task, after_day, block_group)
    if min_blocks is not None:
        rats = rats_with_enough_blocks(select_trials(df, task, after_day), min_blocks)
        trials = trials[trials.animal.isin(rats)]
    return trials


def load_cohorts(after_day=EXPERT_AFTER_DAY, block_group=None, min_blocks=MIN_BLOCKS):
    """{'unstr': uncorrelated rats, 'str': correlated rats}, expert data by default. The csv files are
    read once per Python session. after_day=0 keeps all days."""
    return {task: _select(df, task, after_day, block_group, min_blocks) for task, df in _cohort_tables().items()}


@lru_cache(maxsize=None)
def _unblocked_table():
    return load_sessdf(animals=UNBLOCKED)


def load_groups(names, after_day=EXPERT_AFTER_DAY, block_group=None, min_blocks=MIN_BLOCKS):
    """{name: trials} for groups of utils/cohorts.py GROUPS ('blocked', 'unblocked', 'cued', 'uncued'),
    all in the uncorrelated environment, with the same selection as load_cohorts."""
    groups = {}
    for name in names:
        df = _unblocked_table() if name == 'unblocked' else _cohort_tables()['unstr']
        groups[name] = _select(df[df.animal.isin(GROUPS[name])], 'unstr', after_day, block_group, min_blocks)
    return groups

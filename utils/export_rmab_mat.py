# export sessdf (uncorrelated cohort) and hsb_data (structured cohort) to rMAB-style .mat files
# mirrors collateRatMABData.m, plus a dayOfTraining field
# usage: python utils/export_rmab_mat.py
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dfLoading import read_df, add_block_groups
from utils.trializer_v4 import apply_task_rules_from_json

SESSDF_PATH = 'R:/behavioralData/processed/sessdf-2026-09-25.csv'
HSB_DIR = r"L:\4portProb_structured"
HSB_ANIMALS = ['HSB_F_006', 'HSB_F_007', 'HSB_F_008', 'HSB_F_009', 'HSB_F_010',
               'HSB_F_011', 'HSB_F_012', 'HSB_F_013', 'HSB_F_014']
RULES_PATH = r"C:\Users\dlab\banditAnalysis\utils\task_data.json"
OUT_DIR = r"C:\Users\dlab\OneDrive - Indian Institute of Science\Data\matlab code\data"

UNSTR_ANIMALS = ['test05022023', 'Blissey', 'Chikorita', 'Darkrai', 'Eevee',
                 'Goldeen', 'Hoppip', 'Inkay', 'Jirachi', 'Kirlia', 'Mesprit',
                 'Nidorina', 'Oddish', 'Quilava', 'Raltz', 'Shinx',
                 'Togepi', 'Umbreon', 'Vulpix', 'Wartortle', 'Zacian',
                 'Bayleef', 'Cresselia', 'Emolga', 'Giratina',
                 'Haxorus', 'Ivysaur', 'Jigglypuff', 'Lugia', 'Ninetales', 'Onix',
                 'Pichu', 'Quaxly', 'Sableye', 'Torchic', 'Uxie', 'Vanillish',
                 'Whismur', 'Xerneas', 'Yamper', 'Zorua', 'Azelf', 'Bidoof']

MIN_TRIALS = 50  # minimum trials per session
NARMS = 4
FIELDS = ['ratName', 'weekOfTraining', 'dayOfTraining', 'task', 'rewardprobs', 'narms',
          'trialNums', 'choices', 'rewards', 'blockIDs', 'lesion']


def add_day_task(df):
    df['day_task'] = df.groupby('animal')['datetime'].transform(
        lambda x: ((x - x.min()).dt.days) + 1
    )
    return df


def load_sessdf():
    # same steps as multiAnimalAnalysis.ipynb cell 0
    # unstr only, so day_task counts days within the unstr task (unlike the notebook)
    sessdf = read_df(SESSDF_PATH, task_type=['unstr'])
    sessdf = add_block_groups(sessdf)
    sessdf = sessdf[~(sessdf.rewarded_by_wrong_port == True)]
    sessdf = add_day_task(sessdf)
    sessdf = sessdf[~(sessdf.animal == 'Alakazam')]
    return sessdf


def load_hsb():
    # same steps as multiAnimalAnalysis.ipynb cell 3
    dfs = []
    for an in HSB_ANIMALS:
        df = pd.read_csv(os.path.join(HSB_DIR, an + '.csv'))
        df['animal'] = an
        dfs.append(df)
    hsb_data = pd.concat(dfs).reset_index()
    hsb_data['trial'] = np.arange(hsb_data.shape[0])
    hsb_data['env'] = hsb_data['task']
    hsb_data['task'] = 'str'
    hsb_data['datetime'] = pd.to_datetime(hsb_data.eptime, unit='s')
    hsb_data = hsb_data.rename(columns={'session': 'block_num'})
    hsb_data = add_block_groups(hsb_data)
    hsb_data = hsb_data.rename(columns={'sess_block': 'session'})
    hsb_data['week_task'] = hsb_data.groupby(['animal', 'task'])['datetime'].transform(
        lambda x: ((x - x.min()).dt.days // 7) + 1
    )
    hsb_data = add_day_task(hsb_data)
    hsb_data = apply_task_rules_from_json(hsb_data, rules_path=RULES_PATH)
    return hsb_data


def parse_rewprob(col):
    return np.array([s.strip('[]').split() for s in col], dtype=float)


def valid_rewprob(col):
    # True if every trial has a full NARMS-long reward prob vector
    return col.map(lambda s: len(str(s).strip('[]').split()) == NARMS).all()


def animal_struct(an, group, task, lesion):
    # one struct per animal: rows = sessions, cols = trials (NaN padded)
    sessions = [g for _, g in group.sort_values(['datetime', 'trial']).groupby('session', sort=False)
                if len(g) >= MIN_TRIALS and valid_rewprob(g.rewprobfull)]
    if len(sessions) == 0:
        return None
    nsess = len(sessions)
    maxtrials = max(len(g) for g in sessions)

    choices = np.full((nsess, maxtrials), np.nan)
    rewards = np.full((nsess, maxtrials), np.nan)
    blockIDs = np.full((nsess, maxtrials), np.nan)
    rewardprobs = np.full((nsess, maxtrials, NARMS), np.nan)
    trialNums = np.zeros((nsess, 1))
    weekOfTraining = np.zeros((nsess, 1))
    dayOfTraining = np.zeros((nsess, 1))

    for i, g in enumerate(sessions):
        n = len(g)
        choices[i, :n] = g.port.to_numpy(dtype=float)  # ports already 1-4
        rewards[i, :n] = g.reward.to_numpy(dtype=float)
        blockIDs[i, :n] = g.block_group.to_numpy(dtype=float)  # already 1-indexed
        rewardprobs[i, :n, :] = parse_rewprob(g.rewprobfull)
        trialNums[i, 0] = n
        weekOfTraining[i, 0] = g.week_task.iloc[0]
        dayOfTraining[i, 0] = g.day_task.iloc[0]

    return {'ratName': an, 'weekOfTraining': weekOfTraining, 'dayOfTraining': dayOfTraining,
            'task': task, 'rewardprobs': rewardprobs, 'narms': float(NARMS),
            'trialNums': trialNums, 'choices': choices, 'rewards': rewards,
            'blockIDs': blockIDs, 'lesion': lesion}


def to_struct_array(structs):
    # N x 1 MATLAB struct array (same as cat(1, ...))
    arr = np.empty((len(structs), 1), dtype=[(f, object) for f in FIELDS])
    for i, s in enumerate(structs):
        for f in FIELDS:
            arr[i, 0][f] = s[f]
    return arr


def build(df, task, lesion, animals=None, weeks=None):
    structs = []
    for an, group in df.groupby('animal'):
        if animals is not None and an not in animals:
            continue
        if weeks is not None:
            group = group[group.week_task.isin(weeks)]
        s = animal_struct(an, group, task, lesion)
        if s is not None:
            structs.append(s)
    return to_struct_array(structs)


if __name__ == '__main__':
    stamp = pd.Timestamp.today().strftime('%Y%m%d')

    sessdf = load_sessdf()
    unstr = sessdf[sessdf.task == 'unstr']
    rMABpre = build(unstr, '4ABunstr', 'pre', animals=UNSTR_ANIMALS)
    savemat(os.path.join(OUT_DIR, f'rMABpre_{stamp}.mat'), {'rMABpre': rMABpre}, do_compression=True)
    print(f'rMABpre: {rMABpre.shape[0]} animals')

    hsb_data = load_hsb()
    # reference rMABstr file stores the struct under the variable name 'rMABpre'
    # task rules relabel post-implant sessions as 'ph_implant'; keep pre-implant 'str' only
    rMABstr = build(hsb_data[hsb_data.task == 'str'], '4ABstr', 'pre')
    savemat(os.path.join(OUT_DIR, f'rMABstr_{stamp}.mat'), {'rMABpre': rMABstr}, do_compression=True)
    print(f'rMABstr: {rMABstr.shape[0]} animals')

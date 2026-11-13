# export sessdf (uncorrelated cohort) and hsbdf (correlated cohort) to rMAB-style .mat files
# mirrors collateRatMABData.m, plus a dayOfTraining field
# data are loaded and cleaned as for the paper figures (utils/load_data.py, docs/data_columns.md): whole
# sessions with a wrongly rewarded trial or > MAX_DUPLICATES duplicated rows are dropped; hsbdf keeps
# task 12 ('str') trials before each rat's implant
# usage: python -m utils.export_rmab_mat
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_data import load_hsbdf, load_sessdf

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

    sessdf = load_sessdf(animals=UNSTR_ANIMALS)
    rMABpre = build(sessdf[sessdf.task == 'unstr'], '4ABunstr', 'pre', animals=UNSTR_ANIMALS)
    savemat(os.path.join(OUT_DIR, f'rMABpre_{stamp}.mat'), {'rMABpre': rMABpre}, do_compression=True)
    print(f'rMABpre: {rMABpre.shape[0]} animals')

    hsbdf = load_hsbdf()  # task 12 before the implant only
    # reference rMABstr file stores the struct under the variable name 'rMABpre'
    rMABstr = build(hsbdf, '4ABstr', 'pre')
    savemat(os.path.join(OUT_DIR, f'rMABstr_{stamp}.mat'), {'rMABpre': rMABstr}, do_compression=True)
    print(f'rMABstr: {rMABstr.shape[0]} animals')

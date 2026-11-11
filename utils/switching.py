"""Switching measures per rat, from the shift_t0 / disp columns made by add_shift_info."""
import numpy as np
import pandas as pd

from utils.chance_levels import CHANCE_DISTANCE
from utils.config import PORTS


def add_switch_distance(df):
    """switch_dist: |next port - port| on switch trials, NaN when the rat stayed (notebook cell 103)."""
    return df.assign(switch_dist=df.disp.abs().where(df.shift_t0 == 1))


def by_outcome(df, col, reward):
    """`col` on trials with this outcome (1 rewarded, 0 not) and NaN on the others, for curves split by
    outcome: averaging the result gives the value conditional on the outcome."""
    return df.assign(**{col: df[col].where(df.reward == reward)})


def switch_rate_by_reward(df):
    """P(switch | outcome of the trial) per rat: rats x reward (0, 1) (notebook cells 125-127)."""
    return df.dropna(subset=['reward', 'shift_t0']).groupby(['animal', 'reward']).shift_t0.mean().unstack()


def switch_distance_by_reward(df):
    """Mean switch distance after rewarded and unrewarded trials per rat, and its chance level (cell 113).

    As in calc_dist_metric: the mean |distance| of the switches in each block, then the mean over blocks.
    Chance is CHANCE_DISTANCE averaged over the ports the rat switched from."""
    sw = df[(df.shift_t0 == 1) & df.reward.notna()]
    sw = sw.assign(distance=sw.disp.abs(), chance=sw.port.map(CHANCE_DISTANCE))
    per_block = sw.groupby(['animal', 'reward', 'block_num'])[['distance', 'chance']].mean()
    return per_block.groupby(['animal', 'reward']).mean()


def switch_distance_probs(df, ports=PORTS, by_outcome=True, include_stay=False):
    """P(next port is at distance d | outcome) per rat, over all trials, with chance (notebook cell 137).

    Chance keeps the rat's switch rate and picks the next port at random among the other ports.
    Returns one row per rat, outcome and distance; by_outcome=False ignores the outcome (no reward column).
    include_stay=True adds distance 0, P(stay); its chance is NaN since chance keeps the rat's own stay rate."""
    ports = np.asarray(ports)
    keys = ['animal', 'reward'] if by_outcome else ['animal']
    rows = []
    for key, g in df.dropna(subset=['choice_t1'] + keys[1:]).groupby(keys):
        dist = g.disp.abs().to_numpy()
        start = g.loc[g.shift_t0 == 1, 'port'].to_numpy()
        to_ports = np.abs(ports[None, :] - start[:, None])  # distance from each switch's start port to each port
        if include_stay:
            rows.append((*key, 0, (dist == 0).mean(), np.nan))
        for d in range(1, len(ports)):
            at_d = (to_ports == d).sum(axis=1).mean() / (len(ports) - 1) if len(start) else 0.0
            rows.append((*key, d, (dist == d).mean(), g.shift_t0.mean() * at_d))
    return pd.DataFrame(rows, columns=keys + ['distance', 'p_switch', 'chance'])

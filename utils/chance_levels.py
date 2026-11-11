"""Chance levels drawn as reference lines in the figures."""
import numpy as np

# mean |distance| of a switch from each port when the next port is one of the other three at random
CHANCE_DISTANCE = {1: 2, 2: 4 / 3, 3: 4 / 3, 4: 2}


def chance_reward_rate(df, min_trials=None):
    """Reward rate of random choices: the mean port probability of each block, averaged over each rat's
    blocks and then over rats. min_trials: use only blocks with at least this many trials, as the curves do."""
    blocks = df.groupby(['animal', 'block_num']).agg(n=('port', 'size'), probs=('rewprobfull', 'first'))
    if min_trials is not None:
        blocks = blocks[blocks.n >= min_trials]
    block_mean = blocks.probs.map(lambda s: np.mean(np.array(s.strip('[]').split(), dtype=float)))
    return block_mean.groupby(level='animal').mean().mean() / 100

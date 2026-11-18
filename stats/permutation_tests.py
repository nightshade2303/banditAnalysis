"""Statistical tests, ported from the lab's MATLAB HelperFunctions."""
import numpy as np


def perm_test_2samples(d1, d2, n_perms=10_000, avg_func=np.nanmean, seed=0):
    """Unpaired two-sample permutation test (permTest_2samples.m). The observed statistic is
    |avg_func(d1) - avg_func(d2)|; the null pools both samples and splits them at random into groups of the
    same sizes, n_perms times. p = fraction of null differences strictly greater than the observed one (as in
    the MATLAB code, so p can be 0). Returns (p, observed difference d1 - d2, null differences)."""
    d1, d2 = np.ravel(np.asarray(d1, float)), np.ravel(np.asarray(d2, float))
    d = np.concatenate([d1, d2])
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_perms, d.size)), axis=1)  # one random order of all values per permutation
    null = np.abs(avg_func(d[idx[:, :d1.size]], axis=1) - avg_func(d[idx[:, d1.size:]], axis=1))
    diff = avg_func(d1) - avg_func(d2)
    return float(np.mean(null > abs(diff))), float(diff), null

"""Model-fit diagnostic plots (data vs. simulation), ported from MATLAB.

Typical use (from the repo root):
    from figs.modelFitPlots import load_struct_mat, report_nan_rewards, plot_model_fit
    ub_data = load_struct_mat(r"...\\ub_data.mat")
    summary, locations = report_nan_rewards(ub_data)
    fig1, fig2 = plot_model_fit(ub_data, fit_data="ub", comp_model="myModel")
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.io import loadmat
from scipy.stats import pearsonr

from utils.plotSettings import *  # noqa: F401,F403  (also provides get_parula_cmap)

INT_FIELDS = ("choices", "rewards", "simchoices", "simrewards", "blocks", "simblocks")

DATA_COLOR = "xkcd:cornflower"
SIM_COLOR = "xkcd:pumpkin orange"


# ---------------------------------------------------------------------------
# loading / checks
# ---------------------------------------------------------------------------
def load_struct_mat(path, var_name=None):
    """Load a MATLAB struct array (one entry per rat) as {ratName: {field: array}}.

    INT_FIELDS are cast to int, everything else to float. A field in INT_FIELDS
    that contains NaN is kept as float (NaN can't be stored as int) and a warning
    is issued.
    """
    mat = loadmat(path)
    if var_name is None:
        var_name = next(k for k in mat if not k.startswith("__"))
    records = mat[var_name][0]
    fields = [f for f in records.dtype.names if f != "ratName"]

    data = {}
    for rec in records:
        rat = str(rec["ratName"][0])
        data[rat] = {}
        for f in fields:
            arr = rec[f].astype(float)
            if f in INT_FIELDS:
                if np.isnan(arr).any():
                    warnings.warn(f"{rat}.{f} contains NaN; kept as float")
                else:
                    arr = arr.astype(int)
            data[rat][f] = arr
    return data


def report_nan_rewards(data, reward_key="rewards", block_key="blocks"):
    """Frequency and location of NaN rewards for every rat.

    Returns
    -------
    summary : DataFrame, one row per rat (n_nan, % of trials, sessions affected)
    locations : DataFrame, one row per NaN trial (rat, session, trial, block);
        session/trial are 0-based indices into the (session x trial) arrays.
    """
    rows, locs = [], []
    for rat, d in data.items():
        rw = np.asarray(d[reward_key], dtype=float)
        sess, trial = np.nonzero(np.isnan(rw))
        rows.append({
            "rat": rat,
            "n_nan": len(sess),
            "pct_trials": 100 * len(sess) / rw.size,
            "n_sessions_affected": len(np.unique(sess)),
            "n_sessions": rw.shape[0],
        })
        locs.extend({"rat": rat, "session": s, "trial": t, "block": d[block_key][s, t]}
                    for s, t in zip(sess, trial))
    summary = pd.DataFrame(rows).set_index("rat")
    locations = pd.DataFrame(locs, columns=["rat", "session", "trial", "block"])
    return summary, locations


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _nanmean(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=axis)


def nanste(x, axis=0):
    """Standard error ignoring NaN (std with N-1, as MATLAB's nanstd)."""
    n = np.sum(~np.isnan(x), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n)


def movmean(x, window):
    """Row-wise centred moving mean ignoring NaN, like smoothdata(x, 2, 'movmean', window)."""
    return (pd.DataFrame(x).T
            .rolling(window, center=True, min_periods=1).mean()
            .T.to_numpy())


def get_block_data(data_mat):
    """Direct port of getBlockData.m: drop all-NaN sessions (rows) and shift the
    non-NaN values of each row to the left (order preserved), NaN at the end.

    Kept for reference; split_blocks uses the block-mask version below instead.
    """
    b = data_mat[~np.all(np.isnan(data_mat), axis=1)]
    order = np.argsort(np.isnan(b), axis=1, kind="stable")
    return np.take_along_axis(b, order, axis=1)


def _block_trials(choices, rewards, blocks, block_id, sess_len):
    """Choices/rewards of one block, left-aligned and cut to sess_len trials.

    Like getBlockData, but both arrays are compacted using the *block mask*, so a
    NaN reward stays on its own trial instead of shifting the rewards relative to
    the choices.
    """
    in_block = blocks == block_id
    keep = in_block.any(axis=1)  # sessions that contain this block
    in_block = in_block[keep]
    order = np.argsort(~in_block, axis=1, kind="stable")
    out = []
    for x in (choices, rewards):
        x = np.where(in_block, x[keep].astype(float), np.nan)
        x = np.take_along_axis(x, order, axis=1)
        x = x[:, :sess_len]
        if x.shape[1] < sess_len:
            x = np.pad(x, ((0, 0), (0, sess_len - x.shape[1])), constant_values=np.nan)
        out.append(x)
    return out


def _pad_rows(x, n_rows):
    return np.pad(x, ((0, n_rows - x.shape[0]), (0, 0)), constant_values=np.nan)


def split_blocks(rat_data, choice_key, reward_key, block_key, sess_len, block_ids=(1, 2)):
    """Concatenate the first sess_len trials of each block side by side.

    Returns (choices, rewards), each [n_sessions x len(block_ids)*sess_len].
    """
    per_block = [_block_trials(rat_data[choice_key], rat_data[reward_key],
                               rat_data[block_key], b, sess_len) for b in block_ids]
    # sometimes the second block doesnt have the same num sess
    n_rows = max(c.shape[0] for c, _ in per_block)
    # now make both [n_sessions x 2*sess_len]
    choices = np.hstack([_pad_rows(c, n_rows) for c, _ in per_block])
    rewards = np.hstack([_pad_rows(r, n_rows) for _, r in per_block])
    return choices, rewards


# ---------------------------------------------------------------------------
# behavioural metrics (per rat, arrays are [sessions x trials])
# ---------------------------------------------------------------------------
def switch_metrics(choices, sess_len):
    """Switch indicator and switch distance at every trial (trial 0 is NaN).

    Trials where either the previous or current choice is NaN are NaN, as is the
    first trial of the second block (index sess_len).
    """
    prev, curr = choices[:, :-1], choices[:, 1:]
    valid = ~np.isnan(prev) & ~np.isnan(curr)
    switched = prev != curr

    switches = np.full(choices.shape, np.nan)
    distance = np.full(choices.shape, np.nan)
    switches[:, 1:] = np.where(valid, switched, np.nan)
    # diff betwen prev_choice and current (NaN when the rat stayed)
    distance[:, 1:] = np.where(valid & switched, np.abs(prev - curr), np.nan)
    switches[:, sess_len] = np.nan
    distance[:, sess_len] = np.nan
    return switches, distance


def transition_counts(choices, rewards, n_arms, sess_len):
    """Transition counts choice(t) -> choice(t+1): all trials, after reward, after no reward.

    The transition across the block boundary is excluded.
    """
    prev, curr, prev_rew = choices[:, :-1], choices[:, 1:], rewards[:, :-1]
    valid = ~np.isnan(prev) & ~np.isnan(curr)
    valid[:, sess_len - 1] = False
    valid_rew = valid & ~np.isnan(prev_rew)

    def count(mask):
        tm = np.zeros((n_arms, n_arms))
        np.add.at(tm, (prev[mask].astype(int) - 1, curr[mask].astype(int) - 1), 1)
        return tm

    tm_all = count(valid)
    tm_r = count(valid_rew & (prev_rew > 0))
    tm_nr = count(valid_rew & (prev_rew == 0))
    return tm_all, tm_r, tm_nr


def conditional_switch(choices, rewards, sess_len):
    """Switching split by the previous trial's outcome.

    size(p_switch) = [n_rounds x n_trials x n_rats x 2 (R/NR)]
    (original, not anymore)
    now per rat [n_rounds x n_trials-1 x 2 (NR/R)]

    Returns
    -------
    p_switch : P(switch AND prev outcome) over all trials (joint, not conditional)
    d_switch : switch distance on switch trials, NaN elsewhere
    p_switch_norm : P(switch | prev outcome); NaN on trials of the other outcome,
        i.e. correctly normalized. Not used in the plots yet.
    """
    choice_diff = np.abs(np.diff(choices, axis=1))
    choice_diff[:, sess_len - 1] = np.nan  # transition across the block boundary
    prev_rew = rewards[:, :-1]
    no_rew, rew = prev_rew == 0, prev_rew > 0
    did_switch = choice_diff > 0

    p_switch = np.stack([did_switch & no_rew,  # switch prob conditioned on no-reward
                         did_switch & rew],    # switch prob conditioned on reward
                        axis=2).astype(float)
    d_switch = np.stack([choice_diff * no_rew, choice_diff * rew], axis=2)
    d_switch[d_switch == 0] = np.nan

    # for correcting the normalization of calc sw prob
    valid = ~np.isnan(choice_diff)
    p_switch_norm = np.stack([np.where(valid & no_rew, did_switch, np.nan),  # mask reward trials
                              np.where(valid & rew, did_switch, np.nan)],    # mask no-reward trials
                             axis=2)  # [rounds x trials x 2]
    return p_switch, d_switch, p_switch_norm


def rat_curves(arrays, smooth_win=None, nan_cols=None, axes=(0,)):
    """Per-rat trial curves: mean over `axes` of each array, optional smoothing.

    Returns [n_rats x n_trials]. nan_cols are blanked after smoothing (to break
    the line at the block boundary).
    """
    curves = np.vstack([_nanmean(x, axis=axes) for x in arrays])
    if smooth_win:
        curves = movmean(curves, smooth_win)
    if nan_cols is not None:
        curves[:, nan_cols] = np.nan
    return curves


def rat_means(arrays, cols=None):
    """One number per rat: mean over sessions and trials (optionally only `cols`)."""
    return np.array([_nanmean(x if cols is None else x[:, cols]) for x in arrays])


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------
def shaded_line(ax, curves, color, label):
    """Mean across rats with an SEM band (shadedErrorBar)."""
    m, se = _nanmean(curves, axis=0), nanste(curves, axis=0)
    x = np.arange(1, curves.shape[1] + 1)
    ax.plot(x, m, color=color, lw=1.5, label=label)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.3, lw=0)


def data_vs_sim_lines(ax, dat_curves, sim_curves, ylabel, ylim, legend_loc, fit_data):
    shaded_line(ax, dat_curves, DATA_COLOR, "Data")
    shaded_line(ax, sim_curves, SIM_COLOR, "Sim")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Trials in session")
    ax.set_ylim(ylim)
    if fit_data == "uu":
        ax.set_xlim(0, 100)
    # ax.set_xlim(1, 100)
    ax.grid(True)
    ax.legend(loc=legend_loc, frameon=False, labelcolor="linecolor")


def transition_matrix_plot(ax, tm, title, cmap, norm=None, clim=None):
    im = ax.imshow(tm, cmap=cmap, norm=norm)
    if clim is not None:
        im.set_clim(clim)
    ax.figure.colorbar(im, ax=ax)
    ticks = np.arange(tm.shape[0])
    ax.set_xticks(ticks, ticks + 1)
    ax.set_yticks(ticks, ticks + 1)
    ax.set_title(title)
    ax.set_xlabel("Choice at t+1")
    ax.set_ylabel("Choice at t")


def reward_diff_tm_plot(ax, tm_r, tm_nr, title):
    # normalize each row to get probabilities
    tm_r_norm = tm_r / tm_r.sum(axis=1, keepdims=True)
    tm_nr_norm = tm_nr / tm_nr.sum(axis=1, keepdims=True)
    # subtract
    tm_diff = tm_r_norm - tm_nr_norm
    lo, hi = np.nanmin(tm_diff), np.nanmax(tm_diff)
    # red-white-blue, white at 0 (redwhiteblue(256, 0, [lo, hi]))
    norm = TwoSlopeNorm(vcenter=0, vmin=min(lo, -1e-12), vmax=max(hi, 1e-12))
    transition_matrix_plot(ax, tm_diff, title, cmap="RdBu", norm=norm)


def outcome_bars(ax, sim_vals, dat_vals, ylabel, ylim):
    """Grouped bars (sim, data) x (NR, R) with SEM across rats.

    sim_vals / dat_vals: [n_rats x 2 (NR/R)].
    """
    means = np.vstack([_nanmean(sim_vals, axis=0), _nanmean(dat_vals, axis=0)])  # [group x cond]
    sems = np.vstack([nanste(sim_vals, axis=0), nanste(dat_vals, axis=0)])
    groups = np.arange(2)  # sim, data
    width = 0.4
    colors = [[0, 0, 1], [1, 0, 0]]
    for i in range(2):  # the two conditions
        x_pos = groups + (i - 0.5) * width  # actual centre x of each bar in group i
        ax.bar(x_pos, means[:, i], width, color=colors[i])
        ax.errorbar(x_pos, means[:, i], yerr=sems[:, i], fmt="none",
                    ecolor="k", elinewidth=2, capsize=0)
    ax.set_xticks(groups, ["sim", "data"])
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid(True)


def corr_scatter(ax, x, y, xlabel, ylabel):
    """Per-rat actual vs predicted with a least-squares line and Pearson r."""
    ax.set_box_aspect(1)
    ax.scatter(x, y, color="xkcd:black")
    # Find the coefficients of the linear regression (slope and intercept)
    # remove any NaNs first
    valid = ~np.isnan(x) & ~np.isnan(y)
    coefficients = np.polyfit(x[valid], y[valid], 1)
    # Generate the values of the line of best fit
    xs = np.sort(x[valid])
    ax.plot(xs, np.polyval(coefficients, xs), "-", color="xkcd:grey", label="Line of Best Fit")
    ax.grid(True)
    ax.set_xlabel(xlabel, color=DATA_COLOR)
    ax.set_ylabel(ylabel, color=SIM_COLOR)
    r_val, p = pearsonr(x[valid], y[valid])
    ax.set_title(f"r={r_val:.3f}, p={p:.2e}", fontweight="normal")


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def plot_model_fit(data_str, fit_data, comp_model, sess_len=100, n_arms=4):
    """Data vs. simulation diagnostics for one dataset ('uu', 'ub' or 'sb').

    data_str : {ratName: {field: array}} as returned by load_struct_mat.
    Returns (fig_summary, fig_correlations).
    """
    rats = list(data_str)

    datchoices, datrewards, simchoices, simrewards = [], [], [], []
    for rat in rats:
        c, r = split_blocks(data_str[rat], "choices", "rewards", "blocks", sess_len)
        datchoices.append(c)
        datrewards.append(r)
        # same for sim
        c, r = split_blocks(data_str[rat], "simchoices", "simrewards", "simblocks", sess_len)
        simchoices.append(c)
        simrewards.append(r)

    # switches, transition matrices and outcome-conditioned switching
    datswitches, datdistance = zip(*(switch_metrics(c, sess_len) for c in datchoices))
    simswitches, simdistance = zip(*(switch_metrics(c, sess_len) for c in simchoices))

    datTM_A, datTM_R, datTM_NR = map(sum, zip(*(transition_counts(c, r, n_arms, sess_len)
                                                for c, r in zip(datchoices, datrewards))))
    simTM_A, simTM_R, simTM_NR = map(sum, zip(*(transition_counts(c, r, n_arms, sess_len)
                                                for c, r in zip(simchoices, simrewards))))

    datp_switch, datd_switch, datp_switch_norm = zip(*(conditional_switch(c, r, sess_len)
                                                       for c, r in zip(datchoices, datrewards)))
    simp_switch, simd_switch, simp_switch_norm = zip(*(conditional_switch(c, r, sess_len)
                                                       for c, r in zip(simchoices, simrewards)))

    # plotting the whole thing
    fig1, axes = plt.subplots(2, 5, figsize=(15, 6), layout="constrained")
    tiles = axes.flat  # tiles[0] == MATLAB nexttile(1)

    # reward rate
    datrewards_avg = rat_curves(datrewards, smooth_win=3, nan_cols=sess_len)
    simrewards_avg = rat_curves(simrewards, smooth_win=3, nan_cols=sess_len)
    data_vs_sim_lines(tiles[0], datrewards_avg, simrewards_avg, "Reward rate",
                      (0.35, 0.75), "lower right", fit_data)

    # plot switch rate
    datswitches_avg = rat_curves(datswitches, smooth_win=3, nan_cols=sess_len)
    simswitches_avg = rat_curves(simswitches, smooth_win=3, nan_cols=sess_len)
    data_vs_sim_lines(tiles[1], datswitches_avg, simswitches_avg, "Switch rate",
                      (0, 0.5), "upper right", fit_data)

    # plot sw rate and sw dist conditioned by reward
    wind = 7
    blank = np.arange(sess_len - wind // 2 - 1, sess_len + wind // 2)
    datd_switch_avg = rat_curves(datd_switch, smooth_win=wind, nan_cols=blank, axes=(0, 2))
    simd_switch_avg = rat_curves(simd_switch, smooth_win=wind, nan_cols=blank, axes=(0, 2))
    data_vs_sim_lines(tiles[2], datd_switch_avg, simd_switch_avg, "Switch distance",
                      (0.95, 1.5), "upper right", fit_data)
    tiles[2].axhline(1, ls="--", lw=0.75, color="k")
    # ax.set_xlim(1, 50)
    # ax.set_xlim(0, 100)

    # if you wnt compressibility
    # NOTE: does not run yet -- utils.lzw_helpers.lzw_compress has no `weights`
    # or bias-random-ratio option, which the MATLAB version used.
    # ax = tiles[3]
    # rnd_ratio_sim, rnd_ratio_dat = [], []
    # for choices_all, out in ((simchoices, rnd_ratio_sim), (datchoices, rnd_ratio_dat)):
    #     for rat in range(len(choices_all)):
    #         n_rd = choices_all[rat].shape[0]
    #         rnd_ratio_arr_bl1 = np.zeros(n_rd)
    #         rnd_ratio_arr_bl2 = np.zeros(n_rd)
    #         # (the MATLAB data loop used the *sim* choices for the weights)
    #         sc = simchoices[rat][~np.isnan(simchoices[rat])]
    #         weights = np.histogram(sc, bins=4)[0] / sc.size
    #         for rd in range(n_rd):
    #             sel_a = choices_all[rat][rd, :]
    #             char_choice = "".join(str(int(c)) for c in sel_a[~np.isnan(sel_a)])
    #             kw = dict(return_bias_random_ratio=True, rng="twister",
    #                       n_random_samples=1, weights=weights)
    #             if len(char_choice) < sess_len:
    #                 rnd_ratio_bl1 = lzw_compress(char_choice, **kw)[3]
    #                 rnd_ratio_bl2 = np.nan
    #             else:
    #                 rnd_ratio_bl1 = lzw_compress(char_choice[:sess_len], **kw)[3]
    #                 rnd_ratio_bl2 = np.nan
    #                 if len(char_choice) > 100:
    #                     rnd_ratio_bl2 = lzw_compress(char_choice[sess_len:], **kw)[3]
    #             rnd_ratio_arr_bl1[rd] = rnd_ratio_bl1
    #             rnd_ratio_arr_bl2[rd] = rnd_ratio_bl2
    #         out.append(np.vstack([rnd_ratio_arr_bl1, rnd_ratio_arr_bl2]))
    # rnd_ratio_sim_avg = np.column_stack([_nanmean(x, axis=1) for x in rnd_ratio_sim])  # [2 x n_rats]
    # rnd_ratio_dat_avg = np.column_stack([_nanmean(x, axis=1) for x in rnd_ratio_dat])
    # ax.boxplot(np.vstack([rnd_ratio_sim_avg, rnd_ratio_dat_avg]).T, positions=[1, 2, 3, 4])
    # for i, vals in enumerate(np.vstack([rnd_ratio_sim_avg, rnd_ratio_dat_avg])):
    #     ax.scatter(np.full(vals.size, i + 1), vals, s=30, color="xkcd:grey")
    # ax.set_xticks([1, 2, 3, 4], ["sim bl1", "sim bl2", "data bl1", "data bl2"])
    # ax.set_ylim(bottom=1)
    # ax.set_ylabel("Compressibility")
    # ax.grid(True)

    # otherwise you get TMs
    reward_diff_tm_plot(tiles[3], simTM_R, simTM_NR, "Sim R-NR")
    # real data
    reward_diff_tm_plot(tiles[4], datTM_R, datTM_NR, " Data R-NR")

    # plot transition matrix
    parula = get_parula_cmap()
    transition_matrix_plot(tiles[5], simTM_A / simTM_A.sum(axis=1, keepdims=True),  # row_normalized
                           "Simulation", parula, clim=(0, 0.2))
    transition_matrix_plot(tiles[6], datTM_A / datTM_A.sum(axis=1, keepdims=True),
                           "Data", parula, clim=(0, 0.2))
    tiles[7].axis("off")

    # switch probability by previous outcome (joint probability, uncorrected)
    outcome_bars(tiles[8],
                 np.vstack([_nanmean(x, axis=(0, 1)) for x in simp_switch]),
                 np.vstack([_nanmean(x, axis=(0, 1)) for x in datp_switch]),
                 "p(switch)|R", (0, 0.1))
    # normalized version, P(switch | outcome):
    # outcome_bars(tiles[8],
    #              np.vstack([_nanmean(x, axis=(0, 1)) for x in simp_switch_norm]),
    #              np.vstack([_nanmean(x, axis=(0, 1)) for x in datp_switch_norm]),
    #              "p(switch)|R", (0, 1))

    outcome_bars(tiles[9],
                 np.vstack([_nanmean(x, axis=(0, 1)) for x in simd_switch]),
                 np.vstack([_nanmean(x, axis=(0, 1)) for x in datd_switch]),
                 "d(switch)|R", (1, 1.4))

    fig1.suptitle(f"{fit_data}{comp_model}")

    # plotting correlations between pred and actual
    fig2, axes2 = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    tiles2 = axes2.flat
    cols = slice(0, sess_len) if fit_data == "uu" else None

    corr_scatter(tiles2[0], rat_means(datrewards, cols), rat_means(simrewards, cols),
                 "Actual performance", "Predicted performance")

    # performance index instead
    # 80 - prob of choice taken
    # datregret, simregret = [], []
    # for rat in rats:
    #     d = data_str[rat]
    #     datrewprobs = np.take_along_axis(d["rew_probs"], d["choices"][..., None] - 1, axis=2)[..., 0]
    #     simrewprobs = np.take_along_axis(d["simrps"], d["simchoices"][..., None] - 1, axis=2)[..., 0] * 100
    #     datregret.append((datrewprobs - 10) / (80 - 10))
    #     simregret.append((simrewprobs - 10) / (80 - 10))
    # corr_scatter(tiles2[0], rat_means(datregret), rat_means(simregret),
    #              "Actual performance", "Predicted performance")

    corr_scatter(tiles2[1], rat_means(datswitches, cols), rat_means(simswitches, cols),
                 "Actual switch rate", "Predicted switch rate")
    corr_scatter(tiles2[2], rat_means(datdistance, cols), rat_means(simdistance, cols),
                 "Actual switch distance", "Predicted switch distance")

    # ax = tiles2[3]
    # corr_scatter(ax, rnd_ratio_dat_avg[0], rnd_ratio_sim_avg[0],
    #              "Actual compressibility", "Predicted compressibility")
    tiles2[3].axis("off")

    fig2.suptitle(comp_model)
    return fig1, fig2


def plot_all_datasets(datasets, comp_model, **kwargs):
    """Loop over {'uu': uu_data, 'ub': ub_data, 'sb': sb_data} like the MATLAB `for fitData = datatype`."""
    return {name: plot_model_fit(d, name, comp_model, **kwargs) for name, d in datasets.items()}

import numpy as np
def nLL_ql_gen(x0, data, arms, return_values = False):
    tau, lr_max, lr_sig, decay, sticky, q0 = x0 
    chosen_action = data[0]
    rewarded = data[1]
    block_info = data[2]
    P = np.ones_like(rewarded)*1e-7
    

    rounds = chosen_action.shape[0]
    trials = chosen_action.shape[1]
    ll = 0

    dist_mat = abs(np.arange(arms)[:, None] - np.arange(arms)[None, :])

    # p = np.zeros(shape = (rounds, trials, arms))
    q = np.zeros(shape = (rounds, trials, arms))
    q[0, 0, :] = np.ones(arms)*q0

    for rnd in range(rounds):

        # reset q-val if block-num is restarted in the session
        
        q[rnd, 0, :] = np.ones(arms)*q0
        # upper = 0
        # n_a = np.ones(arms)
        prev_a = np.zeros(arms)

        for t in range(1,trials):
            if np.isnan(chosen_action[rnd, t]):
                break

            # decay q-val
            q[rnd, t, :] = q[rnd, t, :] - (decay*q[rnd, t, :])

            # chosen_action
            a = int(chosen_action[rnd, t]-1)

            # increment n 
            # n_a[a] += 1

            # rewarded?
            r = rewarded[rnd, t]
            
            # determine if locality is involved
            if lr_sig > 0:
                lr = lr_max * np.exp(-(dist_mat[a, :]**2) / (2 * (lr_sig**2)))
                q[rnd, t, :] = q[rnd, t-1, :] + lr*(r - q[rnd, t-1, :])

            else:
                # update q-val
                q[rnd, t, a] = q[rnd, t-1, a] + lr_max*(r - q[rnd, t-1, a])
            
            q[rnd, t, :] = np.clip(q[rnd, t, :], 0, 1)
            # upper = np.sqrt(2*np.log(t+1))/n_a   # UCB-1 (if-needed)

            # calc sticky param
            prev_a = np.zeros(arms)
            prev_a[int(chosen_action[rnd, t-1]-1)] += 1
            
            # calculate softmax
            q_upper = q[rnd, t, :] #+ beta*upper
            p = q_upper/tau + prev_a*sticky
            p = p - np.max(p)
            p = np.exp(p)/np.sum(np.exp(p))
            P[rnd, t] = p[a]

    P[P == 1e-7] = np.nan
    ll += np.nansum(np.log(P))
    nll = -ll
    
    if return_values == True:
        return nll, q

    return nll

def sim_ql_gen(x0, arr_size, arms, return_values = False):
    tau, lr_max, lr_sig, decay, sticky, q0 = x0
    rounds = arr_size[0]
    trials = arr_size[1]
    rng = np.random.default_rng(42)
    dist_mat = abs(np.arange(arms)[:, None] - np.arange(arms)[None, :])
    

    chosen_actions = np.ones(shape = (rounds, trials))*np.nan
    rewarded = np.ones(shape = (rounds, trials))*np.nan

    def fxn(mean, arms, permute = False):
        x = np.linspace(1, arms, arms)
        sig = 1.75/2
        amp = 0.7
        vo = 0.1
        gx = (amp*np.exp(-0.5*((x-mean)**2)/(sig**2)))+vo
        if permute:
            gx = np.random.permutation(gx)
        return gx

    # generate reward probability
    l = [fxn(rng.integers(1, arms+1), arms, True) for _ in range(10000)]

    # p = np.zeros(shape = (rounds, trials, arms))
    q = np.zeros(shape = (rounds, trials, arms))
    q[0, 0, :] = np.ones(arms)*q0

    for rnd in range(rounds):
        rp = l[rnd]
        # reset q-val if block-num is restarted in the session
        
        q[rnd, 0, :] = np.ones(arms)*q0
        # upper = 0
        # n_a = np.ones(arms)
        prev_a = np.zeros(arms)

        for t in range(trials):

            # decay q-val
            q[rnd, t, :] = q[rnd, t, :] - (decay*q[rnd, t, :])

            # calculate softmax
            q_upper = q[rnd, t, :] #+ beta*upper
            p = q_upper/tau + prev_a*sticky
            p = p - np.max(p)
            p = np.exp(p)/np.sum(np.exp(p))
            # print(p)

            # choose action
            actions = np.random.multinomial(1, p)
            chosen_actions[rnd, t] = int(np.arange(4)[actions.nonzero()[0][0]])
            a = int(chosen_actions[rnd, t])
            # print(a)

            # rewarded?
            rewarded[rnd, t] = 1 if rng.uniform() <= rp[a] else 0
            r = rewarded[rnd, t]
            # print(r)

            # determine if locality is involved
            if t<trials-1:
                if lr_sig > 0:
                    lr = lr_max * np.exp(-(dist_mat[a, :]**2) / (2 * (lr_sig**2)))
                    q[rnd, t+1, :] = q[rnd, t, :] + lr*(r - q[rnd, t, :])

                else:
                    # update q-val
                    q[rnd, t+1, a] = q[rnd, t, a] + lr_max*(r - q[rnd, t, a])
                # print(q[rnd, t+1, a])
                q[rnd, t+1, :] = np.clip(q[rnd, t+1, :], 0, 1)
            # upper = np.sqrt(2*np.log(t+1))/n_a   # UCB-1 (if-needed)

            # calc sticky param
            prev_a = np.zeros(arms)
            prev_a[int(chosen_actions[rnd, t])] += 1

    if return_values:
        return chosen_actions, rewarded, q
    return chosen_actions, rewarded

if __name__ == "__main__":

    rounds = 1000
    trials = 100
    arms = 4
    return_values = True
    tau = 0.1
    lr_max = 0.1
    lr_sig = 0.2
    decay = 0.05
    sticky = 1
    q0 = 0.1
    x0 = [tau, lr_max, lr_sig, decay, sticky, q0]
    a, r, q = sim_ql_gen(x0, (rounds, trials), arms, return_values)
    block_info = np.ones_like(r)
    print(f'Peak reward rate {np.mean(r, axis = 0)[trials-1]}')
    bounds = np.array([
        [0.0015, 50], 
        [0, 1], 
        [1e-3, 10], 
        [0, 1], 
        [0, 10],
        [0, 1]
        ])

    from scipy.optimize import minimize
    nll = minimize(
        nLL_ql_gen, 
        x0, 
        ([a, r, block_info], arms), 
        method = 'L-BFGS-B',
        bounds = bounds,
        options = {'disp' : True}
        )




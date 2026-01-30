# policy gradient with gaussian policy
import numpy as np
def f(x, a, b, c):
    '''exponential fxn: a is amplitude, b is decay rate, c is offset'''
    return a*np.exp(b*x) + c

def fxn(mean, arms, permute = False):
    x = np.linspace(1, arms, arms)
    sig = 1.75/2
#     amp = 1/(sig*np.sqrt(2*np.pi))
    amp = 0.7
    vo = 0.1
    gx = (amp*np.exp(-0.5*((x-mean)**2)/(sig**2)))+vo
    if permute:
        gx = np.random.permutation(gx)
    return gx

def simgPG(sessions, trials, a_mu, a_r, arms, a_, b_, c_):
    """
    Gaussian policy gradient for multi-armed bandit problem.
    
    Parameters:
    sessions (int): Number of sessions.
    trials (int): Number of trials per session.
    a_mu (float): Learning rate for action update.
    a_r (float): Learning rate for reward update.
    arms (int): Number of arms.
    a_ (float): Amplitude for the exponential function.
    b_ (float): Decay rate for the exponential function.
    c_ (float): Offset for the exponential function.
    
    Returns:
    mu, sigma, V, rp, r, a: numpy arrays containing the action values, state values, rewards, and actions.
    """
    p = np.zeros((arms, sessions, trials))
    a = np.zeros((sessions, trials))
    r = np.zeros((sessions, trials))
    rp = np.zeros((sessions, arms))
    mu = np.zeros((sessions, trials))
    V = np.zeros((sessions, trials))
    sigma = np.ones((sessions, trials))

    for s in np.arange(sessions):
        center = np.random.choice(np.arange(1, arms+1))
        rp[s] = fxn(center, arms, True)
        mu[s, 0] = np.random.choice(np.arange(1, arms+1))
        V[s, 0] = 0.25
        sigma[s, 0] = 0.5 

        for t in np.arange(trials):
            # calculate probability of actions
            p[:, s, t] = np.array([np.exp(-(i - mu[s, t])**2/(2*(sigma[s, t]**2))) for i in np.arange(1, arms+1)])
            p[:, s, t] = p[:, s, t]/np.sum(p[:, s, t])

            # sample action
            actions = np.random.multinomial(1, p[:, s, t])
            a[s, t] = np.arange(arms, dtype = int)[actions.nonzero()[0][0]]+1

            # get reward 
            rand = np.random.uniform(0, 1)
            r[s, t] = 1 if rand <= rp[s][int(a[s, t]) - 1] else 0

            # reward prediction error
            delta = r[s, t] - V[s, t]
            if t<trials-1:
                # action update
                mu[s, t+1] = mu[s, t] + (a_mu*delta*(a[s, t] - mu[s, t]))

                # calculate state value
                V[s, t+1] = V[s, t] + a_r*delta

                # use state value as sigma?
                # sigma[t+1] = np.exp(-V[t+1]*0.9)
                # sigma[s, t+1] = f(V[s, t+1], a_, b_, c_)
                sigma[s, t+1] = f(V[s, t+1], a_, b_, c_)
    
    return mu, sigma, V, rp, r, a

def fitgPG(x0, sessdf, arms):
    a_mu, a_r, a_, b_, c_, mu_init, V_init, sigma_init = x0
    ll = 0
    
    sessions = sessdf['session'].nunique()
    trials = 100 # automate later
    p = np.zeros((arms, sessions, trials))
    mu = np.zeros((sessions, trials))
    V = np.zeros((sessions, trials))
    sigma = np.ones((sessions, trials))
    P = np.zeros((sessions, trials))
    
    for s, (_, group) in enumerate(sessdf.reset_index().groupby('session')):
        mu[s, 0] = mu_init
        V[s, 0] = V_init
        sigma[s, 0] = sigma_init
        for t, trial in group.iterrows():
            p[:, s, t] = np.array([np.exp(-(i - mu[s, t])**2/(2*(sigma[s, t]**2))) for i in np.arange(1, arms+1)])
            p[:, s, t] = p[:, s, t]/np.sum(p[:, s, t])

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of selected action on this trial
            P[s, t] = p[index, s, t]

            # rewarded?
            r = trial['reward']

            # reward prediction error
            delta = r - V[s, t]
            if t<trials:
                # action update
                mu[s, t+1] = mu[s, t] + (a_mu*delta*(a - mu[s, t]))

                # calculate state value
                V[s, t+1] = V[s, t] + a_r*delta

                # use state value as sigma?
                # sigma[t+1] = np.exp(-V[t+1]*0.9)
                # sigma[s, t+1] = f(V[s, t+1], a_, b_, c_)
                sigma[s, t+1] = f(V[s, t+1], a_, b_, c_)

    ll += np.nansum(np.log(P))
    nll = -ll
    return nll
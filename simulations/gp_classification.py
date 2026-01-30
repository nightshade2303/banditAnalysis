import numpy as np
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF



def softmax(arr, temp):
    P = np.exp(arr*1/temp)
    P = P/sum(P)
    return P


# my function, with permuted probabilities
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

################ GIVING REWARD #####################
def rewarding(prob, reward_val):
    temp = reward_val
    rand = np.random.uniform(0, 1)
    return temp if rand <= prob else 0

# hyperparameters (need to make a class object later that receives these)
trials = 100
sessions = 1000
arms = 4
ls = 1.13
n_samples = 500
be = 1
temp = 0.2032
rp_set = [fxn(np.random.randint(1, arms+1), arms, True) for i in range(10000)]

def gp_classifier(sessions, trials, arms, x0):

    a = np.zeros((sessions, trials))
    r = np.zeros((sessions, trials))
    p_est = np.zeros((sessions, trials, arms), dtype = float)
    
    X = np.arange(arms).reshape(-1, 1) # excuse sklearn for using this dumb syntax
    for sess in range(sessions):
        a, r, p_est = run_session(sess, trials, x0)
    return a, r, p_est

def run_session(sess, trials, x0):
    # reward probability changes here
    rp = rp_set[sess]
    # reset model
    gp = GaussianProcessClassifier(kernel=RBF(length_scale=ls), optimizer = None)

    for t in range(trials):
        if len(np.unique(r[sess, :(t+1)])) > 1:
            # update gp using all the info we got so far on actions and rew
            gp.fit(X[a[sess, :(t+1)].astype(int)], r[sess, :(t+1)])

            # draw arm using UCB
            # get latent mean and variance
            mu, var = gp.latent_mean_and_variance(X)
            prob_mean = expit(mu)
            prob_std = expit(mu + np.sqrt(var))
            
            # sample from this latent process
            # samples = np.random.normal(mu[:, None], np.sqrt(var)[:, None], size=(len(mu), n_samples))
            # prob_samples = expit(samples)
            # # shoulve been the true output of the gp but whatever
            # prob_mean = prob_samples.mean(axis=1)
            # prob_std = prob_samples.std(axis=1)
            # compute UCB
            p_est[sess, t, :] = softmax(prob_mean + be*prob_std, temp = temp)
            # select arm using this prob
            a[sess, t] = np.random.multinomial(1, p_est[sess, t, :]).nonzero()[0][0]
            chosen = int(a[sess, t])

            # reward chosen arm
            r[sess, t] = rewarding(rp[chosen], 1)

        else:
            a[sess, t] = np.random.choice(np.arange(arms))
            chosen = int(a[sess, t])
            r[sess, t] = rewarding(rp[chosen], 1)

    return a, r, p_est


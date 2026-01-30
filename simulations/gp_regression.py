import numpy as np


from sklearn.gaussian_process import GaussianProcessRegressor
trials = 100
sessions = 1000
arms = 4
ls = 1
n_samples = 500
be = 1
temp = 1

a = np.zeros((sessions, trials))
r = np.zeros((sessions, trials))
p_est = np.zeros((sessions, trials, arms), dtype = float)
rp_set = [fxn(np.random.randint(1, arms+1), arms, True) for i in range(10000)]
X = np.arange(N).reshape(-1, 1) # excuse sklearn for using this dumb syntax

for sess in range(sessions):
    # reward probability changes here
    rp = rp_set[sess]
    # reset model?
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=ls), alpha = 1, optimizer=None)

    for t in range(trials):
        if len(np.unique(r[sess, :(t+1)])) > 1:
            # update gp using all the info we got so far on actions and rew
            gp.fit(X[a[sess, :(t+1)].astype(int)], r[sess, :(t+1)])

            # draw arm using UCB
            # get latent mean and variance
            mu, sd = gp.predict(X, return_std = True)
            # sample from this latent process
            # samples = np.random.normal(mu[:, None], np.sqrt(var)[:, None], size=(len(mu), n_samples))
            # prob_samples = expit(samples)
            # # shoulve been the true output of the gp but whatever
            # prob_mean = prob_samples.mean(axis=1)
            # prob_std = prob_samples.std(axis=1)
            # compute UCB
            p_est[sess, t, :] = softmax(mu + be*sd, temp = temp)
            # select arm using this prob
            a[sess, t] = np.random.multinomial(1, p_est[sess, t, :]).nonzero()[0][0]
            chosen = int(a[sess, t])

            # reward chosen arm
            r[sess, t] = rewarding(rp[chosen], 1)

        else:
            a[sess, t] = np.random.choice(np.arange(arms))
            chosen = int(a[sess, t])
            r[sess, t] = rewarding(rp[chosen], 1)


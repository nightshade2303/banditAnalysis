import numpy as np
# all the different models here::
def nllQlearning(x0, sessdf, arms):
    alpha, tau = x0
    ll = 0
    
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*q)
            P = P/ np.sum(P)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of selected action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']

            # compute q value
            q[index] = q[index] + alpha*(r - q[index])
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllstickyQlearning(x0, sessdf, arms):
    alpha, tau, sticky = x0
    ll = 0
    
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        h = np.zeros(arms)
        q = np.ones(arms)/arms

        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q+h))
            P = P/ np.sum(P)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']

            # compute q value
            q[index] = q[index] + alpha*(r - q[index])

            # add persevarative term to chosen arm
            chosen = np.zeros(arms)
            chosen[index] = 1
            h = h + sticky*(chosen - h)
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllforgetQlearning(x0, sessdf, arms):
    alpha_c, alpha_uc, tau = x0
    ll = 0
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        q = np.ones(arms)/arms
        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q))
            P = P/ np.sum(P)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']

            # compute q value - update all arms with respective alpha dep. on chosen arm
            for val in range(len(q)):
                if val == index:
                    q[val] = q[val] + alpha_c*(r - q[index])
                else:
                    q[val] = q[val] + alpha_uc*(r - q[index])
            
            # if any q value is < 0 make it 0
            q[q<0] = 0
            q[q>1] = 1

    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllstickyforgetQlearning(x0, sessdf, arms):
    alpha_c, alpha_uc, tau, sticky = x0
    ll = 0
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        q = np.ones(arms)/arms
        h = np.zeros(arms)
        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q))
            P = P/ np.sum(P)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']

            # compute q value - update all arms with respective alpha dep. on chosen arm
            for val in range(len(q)):
                if val == index:
                    q[val] = q[val] + alpha_c*(r - q[index])
                else:
                    q[val] = q[val] + alpha_uc*(r - q[index])
            
            # add persevarative term to chosen arm
            chosen = np.zeros(arms)
            chosen[index] = 1
            h = h + sticky*(chosen - h)

            # if any q value is < 0 make it 0
            q[q<0] = 0
            q[q>1] = 1
    
    ll += np.nansum(np.log(p))     
    nll = -ll
    return nll

def nllmatQlearning(x0, sessdf, arms):

    alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag, tau= x0
    ll = 0
    alpha = np.array([[alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag],
                      [alpha_1diag, alpha_diag, alpha_1diag, alpha_2diag],
                      [alpha_2diag, alpha_1diag, alpha_diag, alpha_1diag],
                      [alpha_3diag, alpha_2diag, alpha_1diag, alpha_diag]])
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        q = np.ones(arms)/arms
        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*q)
            if np.sum(np.isinf(P))>=1:
                print(q)
                print(tau)
            P = P/ np.sum(P)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']
            
            # compute q value - update all arms with respective alpha dep. on chosen arm
            q = np.array([q[i] + ((alpha[index, i])*(r - q[index])) for i in range(len(q))])
            # if any q value is < 0 make it 0
            q[q<0] = 0
            q[q>1] = 1
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllmatQlearning_sticky(x0, sessdf, arms):
    alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag, tau, sticky = x0
    ll = 0
    alpha = np.array([[alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag],
                      [alpha_1diag, alpha_diag, alpha_1diag, alpha_2diag],
                      [alpha_2diag, alpha_1diag, alpha_diag, alpha_1diag],
                      [alpha_3diag, alpha_2diag, alpha_1diag, alpha_diag]])
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        h = np.zeros(arms)
        q = np.ones(arms)/arms
        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q+h))
            P = P/ np.sum(P)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']

            # compute q value - update all arms with respective alpha dep. on chosen arm
            q = np.array([q[i] + ((alpha[index, i])*(r - q[index])) for i in range(len(q))])

            # add persevarative term to chosen arm
            chosen = np.zeros(arms)
            chosen[index] = 1
            h = h + sticky*(chosen - h)
            
            # if any q value is < 0 make it 0
            q[q<0] = 0
            q[q>1] = 1

    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllmatQlearning_stickyscaled(x0, sessdf, arms):
    alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag, tau, sticky, H = x0
    ll = 0
    alpha = np.array([[alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag],
                      [alpha_1diag, alpha_diag, alpha_1diag, alpha_2diag],
                      [alpha_2diag, alpha_1diag, alpha_diag, alpha_1diag],
                      [alpha_3diag, alpha_2diag, alpha_1diag, alpha_diag]])
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        h = np.zeros(arms)
        q = np.ones(arms)/arms
        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q+(H*h)))
            P = P/ np.sum(P)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']

            # compute q value - update all arms with respective alpha dep. on chosen arm
            q = np.array([q[i] + ((alpha[index, i])*(r - q[index])) for i in range(len(q))])

            # add persevarative term to chosen arm
            chosen = np.zeros(arms)
            chosen[index] = 1
            h = h + sticky*(chosen - h)
            
            # if any q value is < 0 make it 0
            q[q<0] = 0
            q[q>1] = 1

    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllmatQlearning_stickymat(x0, sessdf, arms):
    alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag, tau, sticky_diag, sticky_1diag, sticky_2diag, sticky_3diag = x0
    ll = 0
    alpha = np.array([[alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag],
                      [alpha_1diag, alpha_diag, alpha_1diag, alpha_2diag],
                      [alpha_2diag, alpha_1diag, alpha_diag, alpha_1diag],
                      [alpha_3diag, alpha_2diag, alpha_1diag, alpha_diag]])
    sticky = np.array([[sticky_diag, sticky_1diag, sticky_2diag, sticky_3diag],
                       [sticky_1diag, sticky_diag, sticky_1diag, sticky_2diag], 
                       [sticky_2diag, sticky_1diag, sticky_diag, sticky_1diag], 
                       [sticky_3diag, sticky_2diag, sticky_1diag, sticky_diag]])
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        h = np.zeros(arms)
        q = np.ones(arms)/arms
        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q+h))
            if np.sum(np.isinf(P))>=1:
                print(q)
                print(h)
                print(sticky)
                print(tau)
            P = P/ np.sum(P)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']

            # compute q value - update all arms with respective alpha dep. on chosen arm
            q = np.array([q[i] + ((alpha[index, i])*(r - q[index])) for i in range(len(q))])

            # add persevarative term to chosen arm
            chosen = np.zeros(arms)
            chosen[index] = 1
            h = np.array([h[i] + ((sticky[index, i])*(chosen[index] - h[index])) for i in range(len(h))])
            
            # if any q value is < 0 make it 0
            q[q<0] = 0
            q[q>1] = 1

            # if any h value is < 0 make it 0
            h[h<0] = 0
            h[h>1] = 1

    ll += np.nansum(np.log(p))
    nll = -ll
    return nll
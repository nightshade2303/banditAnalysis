from opconNosepokeFunctions import *
from supplementaryFunctions import *
from scipy.optimize import minimize

def nllEpsGreedy(x0, sessdf, arms):
    alpha, eps = x0
    ll = 0
    
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        for ind, trial in group.iterrows():
            
            # find max q value
            maxq, = np.where(q==np.amax(q))
            lowq, = np.where(q<np.amax(q))
            
            # fill P array
            P = np.ones(arms)*(eps/arms)
            P[maxq] = (1-eps)/len(maxq)
            if len(lowq)==0:
                P = np.ones(arms)/arms
            else:
                P[lowq] = eps/len(lowq)

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]
#             print(P, index, np.where(q == np.amax(q)), sum(P))

            # rewarded?
            r = trial['reward']

            # compute q value
            q[index] = q[index] + alpha*(r - q[index])
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

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

            # probability of each action on this trial
            p[ind] = P[index]
            
            # rewarded?
            r = trial['reward']

            # compute q value
            q[index] = q[index] + alpha*(r - q[index])
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllWeightedBiasQlearning(x0, sessdf, arms):
    alpha, tau, w, bias = x0
    bias = np.array(bias)
    ll = 0
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        for ind, trial in group.iterrows():

            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q))
            P = P/ np.sum(P)
            
            # weights and bias
            P = w*P+ (1-w)*bias
            
            # which action on this trial
            a = trial['port']
            index = int(a-1)
            
            # probability of each action on this trial
            p[ind] = P[index]

            # rewarded?
            r = trial['reward']

            # compute q value
            q[index] = q[index] + alpha*(r - q[index])
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllWeightedBiasQlearningAllP(x0, sessdf, arms):
    alpha, tau, w, bias_arm1, bias_arm2, bias_arm3, bias_arm4 = x0
    
    ll = 0
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        for ind, trial in group.iterrows():
            
            #bias = [bias_arm1, bias_arm2, bias_arm3, bias_arm4] 
            bias = np.array([bias_arm1, bias_arm2, bias_arm3, bias_arm4])
            bias = bias/np.sum(bias)
            
            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q))
            P = P/ np.sum(P)
            
            # weights and bias
            P = w*P+ (1-w)*bias
            
            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of each action on this trial
            p[ind] = P[index]

            # rewarded?
            r = trial['reward']

            # compute q value
            q[index] = q[index] + alpha*(r - q[index])
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllBiasQlearningAllP(x0, sessdf, arms):
    alpha, tau, bias_arm1, bias_arm2, bias_arm3, bias_arm4 = x0
    
    ll = 0
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        for ind, trial in group.iterrows():
            
            #bias = [bias_arm1, bias_arm2, bias_arm3, bias_arm4] 
            bias = np.array([bias_arm1, bias_arm2, bias_arm3, bias_arm4]) 
            
            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.exp(invtemp*(q+bias))
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
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllQ0Qlearning(x0, sessdf, arms):
    alpha, tau, q0_arm1, q0_arm2, q0_arm3, q0_arm4 = x0
    
    ll = 0
    p = np.zeros(len(sessdf))
    q = np.array([q0_arm1, q0_arm2, q0_arm3, q0_arm4])   #q0 is a free param here
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
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

            # compute q value
            q[index] = q[index] + alpha*(r - q[index])
            
    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def nllAddBiasQlearning(x0, sessdf, arms):
    alpha, tau, bias_arm1, bias_arm2, bias_arm3, bias_arm4 = x0
    
    ll = 0
    p = np.zeros(len(sessdf))
    q = np.ones(arms)/arms
    
    for sessnum, group in sessdf.reset_index().groupby('session'):
        for ind, trial in group.iterrows():
            
            #bias = [bias_arm1, bias_arm2, bias_arm3, bias_arm4] 
            bias = np.array([bias_arm1, bias_arm2, bias_arm3, bias_arm4]) 
            
            # softmax prob of choosing actions
            invtemp=1/tau
            P = np.sum([np.exp(invtemp*(q)),bias], axis = 0)
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
            
    ll += np.nansum(np.log(p))
    nll = -ll
    
    return nll

def process_session(ind, box, d, bounds, arms, trialsinsess, model):
    filtered = d.groupby('session').filter(lambda x: x.reward.size >= trialsinsess)
    args = (filtered, arms)
    
    # parameters
    params = np.zeros(len(bounds))
    
    # random seed generation for each process
    import time
    
    # randomize parameters given bounds
    for i, bound in enumerate(bounds):
        (bound_low, bound_high) = bound
        np.random.seed(int(str(time.time_ns())[12:]))
        params[i] = np.random.uniform(bound_low, bound_high)
    
    # print params 
    print(f' initializing with --- {params}')
    
    result = minimize(model,
                      x0=params,
                      args=args,
                      method='Nelder-Mead',
                      bounds=bounds,
                      options={'maxiter': 1000, 'disp': True})
    return box, result.fun, result.x
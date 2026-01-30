import numpy as np
import pandas as pd
def gen_qvalues(dataset, params, arms = 4, mode = 'vanilla'):
    dataset['all_p_qvalue'] = np.zeros(dataset.shape[0])
    dataset['all_qvalue'] = np.zeros(dataset.shape[0])
    dataset[['all_p_qvalue', 'all_qvalue']]=dataset[['all_p_qvalue', 'all_qvalue']].astype('object')
    
    # compute q val for fixed params

    p = np.zeros(dataset.shape[0])
    q = np.ones(arms)/arms
    
    if mode=='vanilla':
        
        alpha, tau = params
        for sessnum, group in dataset.groupby('session#'):
            q = np.ones(arms)/arms

            for ind, trial in group.iterrows():

                # softmax prob of choosing actions
                invtemp=1/tau
                P = np.exp(invtemp*q)
                P = P/ np.sum(P)

                # which action on this trial
                a = trial['port']
                index = int(a-1)

                # probability of each action on this trial
    #             dataset['all_p_qvalue'].loc[ind] = list(P)

                # rewarded?
                r = trial['reward']

                # compute q value
                q[index] = q[index] + alpha*(r - q[index])
                dataset['all_qvalue'].loc[ind] = list(q)
        ll += np.nansum(np.log(p))
        nll = -ll
                
    elif mode=='biased':
        alpha, tau, bias_arm1, bias_arm2, bias_arm3, bias_arm4 = x0
    
        ll = 0
        p = np.zeros(len(sessdf))
        q = np.ones(arms)/arms

        for sessnum, group in sessdf.reset_index().groupby('session#'):
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
                dataset['all_qvalue'].loc[ind] = list(q)
        ll += np.nansum(np.log(p))
        nll = -ll
            
    return dataset, ll
# imports
import numpy as np
import pandas as pd
from qlfittingFunctions_20250319 import nllQlearning

# data - load 5 models. reserve 20% sessions for testing nll
def load_data_into_dict(model_num, seed_n = 42):
    data_path = fr"L:\4_armed_bandit_metaRL\Unstructured_net\modelnum_{model_num}\test\4"
    actions = np.load(data_path+r"\episodic_actions.npy")
    reward = np.load(data_path+r"\episodic_rewards.npy")

    np.random.seed(seed_n)

    # make dataframe which is correctly shaped
    sessdf = pd.DataFrame({'port':(actions+1).flatten(), 'reward':reward.flatten()})
    sessdf['session'] = np.repeat(range(0, actions.shape[0]), actions.shape[1])
    sessdf = sessdf[sessdf.session < 200]
    test_sess = np.random.choice(sessdf.session.unique(), int(sessdf.session.nunique()*0.2), replace = False)
    train_data = sessdf[~sessdf.session.isin(test_sess)]
    return train_data, test_sess

# vanilla Q function
from pybads import BADS
extra_params = 4
n_optim = 5
k = 2 # number of parameters for bic

options = {}
results = {}
for rnn in range(1, 6):
    data, test_sess = load_data_into_dict(rnn)
    n_trials = data.shape[0]
    
    def fun_for_pybads(x):
        return nllQlearning(x, data, extra_params)
    for n in range(n_optim):
        # run multiple optimizations
        print('Running optimization ' + str(n) + '...')
        options['random_seed'] = n
        bads = BADS(fun_for_pybads, None, [0.0, 0.0015], [1.0, 100], options=options)
        # kept changing bounds so something would atleast work lol
        optimize_result = bads.optimize()
        results[rnn] = [optimize_result.x, optimize_result.fval, optimize_result.success]
        nll = optimize_result.fval
        bic = k*np.log(n_trials) + 2*nll
         # open a file and store results
        with open(fr'L:\4_armed_bandit_metaRL\Unstructured_net\vanillaql.csv', 'a') as f:
            f.write(f'{rnn},{n},{results[rnn][0]},{results[rnn][1]},{results[rnn][2]},{test_sess},{bic}\n')

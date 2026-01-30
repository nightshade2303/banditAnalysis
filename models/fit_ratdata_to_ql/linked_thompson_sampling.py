from qlfittingFunctions import *
from temp_utils.opconNosepokeFunctions import *
from temp_utils.supplementaryFunctions import *

def nllThompson(x0, df, arms):
    alpha0, beta0, increment = x0
    ll = 0
    p = np.zeros(len(df))

    for sessnum, group in df.reset_index().groupby('session'):
        alphas = np.zeros(arms)
        betas = np.zeros(arms)
        for ind, trial in group.iterrows():

            # draw expectation of each arm being selected (alp/ alp+beta)
            # alternatively, draw some samples from the b distribution to estimate the expected value of each arm
            P = [(alphas[arm]+alpha0/(alphas[arm]+alpha0+betas[arm]+beta0)) for arm in range(arms)]

            # which action on this trial
            a = trial['port']
            index = int(a-1)

            # probability of selected action on this trial
            p[ind] = P[index]

            # rewarded?
            r = trial['reward']

            # increment alphas and betas
            alphas[index] += increment*r
            betas[index] += increment*(1-r)

    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def load_data_into_dict(seed_n = 42):
# load semi-processed df from pickle
    import pickle 
    with open('L:/4portProb_processed/cleandf.pkl', 'rb') as f:
        df = pickle.load(f)

    # subset for expert performance
    subset_df = df[(df.sess_bin>=4) & (df.task == 'unstr')]
    from temp_utils.dfLoading import subset_trials, add_block_groups
    
    # add block groups and select only first block 
    subset_df = add_block_groups(subset_df)
    subset_df = subset_df[subset_df.block_group==1]
    trialsinsess = 100
    # then subset for first 100 trials
    subset_df = subset_trials(subset_df, trialsinsess=trialsinsess, head_trials = trialsinsess)

    np.random.seed(seed_n)

    test_sess_dict = {}
    # pick 10% sessions for testing
    for animal in subset_df.animal.unique():
        unique_sess = subset_df[subset_df.animal == animal].session.nunique()
        test_sess = np.random.choice(subset_df[subset_df.animal == animal].session.unique(), int(unique_sess*0.2), replace = False)
        test_sess_dict[animal] = test_sess
        subset_df.drop(subset_df[(subset_df.animal == animal) & (subset_df.session.isin(test_sess))].index, inplace = True)

    sess_dict = {key: group for key, group in subset_df.groupby('animal')}

    print('loaded data')
    return sess_dict, test_sess_dict

from pybads import BADS
animals = ['test05022023','Blissey', 'Chikorita', 'Darkrai', 'Eevee',
            'Goldeen', 'Hoppip', 'Inkay', 'Jirachi',
            'Kirlia', 'Nidorina', 'Phione', 'Quilava', 'Raltz', 'Togepi',
            'Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian']
sess_dict, test_sess_dict = load_data_into_dict()
extra_params = 4
n_optim = 5
k = 5 # number of parameters for bic

options = {}
results = {}
for animal in animals:
    data = sess_dict[animal]
    n_trials = data.shape[0]
    
    def fun_for_pybads(x):
        return nllThompson(x, data, extra_params)
    for n in range(n_optim):
        # run multiple optimizations
        print('Running optimization ' + str(n) + '...')
        options['random_seed'] = n
        bads = BADS(fun_for_pybads,
                    None,
                    [0.5, 0.5, 0],
                    [10, 10, 1], 
                    options = options)
        # kept changing bounds so something would atleast work lol
        optimize_result = bads.optimize()
        results[animal] = [optimize_result.x, optimize_result.fval, optimize_result.success]
        nll = optimize_result.fval
        bic = k*np.log(n_trials) + 2*nll
         # open a file and store results
        with open('L:/4portProb_simulations/model_parameters_20250816/thompson.csv', 'a') as f:
            f.write(f'{animal},{n},{results[animal][0]},{results[animal][1]},{results[animal][2]},{test_sess_dict[animal]},{bic}\n')

import numpy as np
from temp_utils.supplementaryFunctions import tic, toc
def nllThompson(x0, df, arms):
    alpha0, beta0 = x0
    ll = 0
    sessions = df.session.value_counts()
    chosen_action = df.port.to_numpy()
    rewarded = df.reward.to_numpy()
    p = np.zeros(len(df))
    session_ends = np.cumsum(sessions)
        
    sess_num = 0
    sess_start = True
    for trial in range(len(chosen_action)):
        # check if sess restarted
        if sess_start == True:
            # if group.block_group.unique()[0] == 1:
            alphas = np.ones(arms)*alpha0
            betas = np.ones(arms)*beta0
        sess_start = False

        # draw expectation of each arm being selected (alp/ alp+beta)
        # alternatively, draw some samples from the b distribution to estimate the expected value of each arm
        samples = np.random.beta(alphas, betas, size=(500, arms))

        # draw arm using samples drawn from distr
        arm_choices = np.argmax(samples, axis=1)
        counts = np.bincount(arm_choices, minlength=arms)
        P = counts / counts.sum()
        P = np.clip(P, a_min=1e-6, a_max = 1)

        # which action on this trial
        a = chosen_action[trial]
        chosen = int(a-1)

        # probability of selected action on this trial
        p[trial] = P[chosen]

        # rewarded?
        r = rewarded[trial]

        # increment alphas and betas
        alphas[chosen] += r+alpha0
        betas[chosen] += (1-r)+beta0

        # check how many trials elapsed
        if trial == session_ends.iloc[sess_num]:
            sess_start=True
            sess_num += 1

    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def load_data_into_dict(seed_n = 42):
# load semi-processed df from pickle
    import pickle 
    with open('C:/Users/dlab/Downloads/cleandf.pkl', 'rb') as f:
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

from scipy.optimize import differential_evolution
animals = ['test05022023','Blissey', 'Chikorita', 'Darkrai', 'Eevee',
            'Goldeen', 'Hoppip', 'Inkay', 'Jirachi',
            'Kirlia', 'Nidorina', 'Phione', 'Quilava', 'Raltz', 'Togepi',
            'Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian']
sess_dict, test_sess_dict = load_data_into_dict()
extra_params = 4
n_optim = 5
k = 2 # number of parameters for bic

options = {}
results = {}
filepath = 'L:/4portProb_simulations/model_parameters_20250826/thompson_first_block_de'
tic()
for animal in animals:
    data = sess_dict[animal]
    n_trials = data.shape[0]
    
    for n in range(n_optim):
        # run multiple optimizations
        print('Running optimization ' + str(n) + '...')
        
        optimize_result = differential_evolution(
            func = nllThompson,
            bounds = ([1, 10], [1, 10]),
            args = (data, extra_params),
            disp = True
        )

        results[animal] = [optimize_result.x, optimize_result.fun, optimize_result.success]
        nll = optimize_result.fun
        bic = k*np.log(n_trials) + 2*nll
         # open a file and store results
        with open(f'{filepath}.csv', 'a') as f:
            f.write(f'{animal},{n},{results[animal][0]},{results[animal][1]},{results[animal][2]},{test_sess_dict[animal]},{bic}\n')
with open(f'{filepath}.log', 'a') as f:
    f.write(toc())
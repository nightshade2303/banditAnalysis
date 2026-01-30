from temp_utils.supplementaryFunctions import tic, toc
import numpy as np

def nllmatQlearning(x0, df, arms):
    alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag, tau, sticky, H = x0
    ll = 0
    alpha = np.array([[alpha_diag, alpha_1diag, alpha_2diag, alpha_3diag],
                      [alpha_1diag, alpha_diag, alpha_1diag, alpha_2diag],
                      [alpha_2diag, alpha_1diag, alpha_diag, alpha_1diag],
                      [alpha_3diag, alpha_2diag, alpha_1diag, alpha_diag]])
    p = np.zeros(len(df))
    q = np.ones(arms)/arms
    h = np.zeros(arms)
    sessions = df.session.value_counts()
    chosen_action = df.port.to_numpy()
    rewarded = df.reward.to_numpy()
    session_ends = np.cumsum(sessions)
        
    sess_num = 0
    sess_start = True
    
    for trial in range(len(chosen_action)):
        # check if sess restarted
        if sess_start == True:
            # if group.block_group.unique()[0] == 1:
            q = np.ones(arms)/arms
            h = np.zeros(arms)
        sess_start = False

        # softmax prob of choosing actions
        invtemp=1/tau
        P = np.exp((invtemp*q)+(H*h))
        P = P/ np.sum(P)

        # which action on this trial
        a = chosen_action[trial]
        index = int(a-1)

        # probability of each action on this trial
        p[trial] = P[index]
        
        # rewarded?
        r = rewarded[trial]

        # compute q value - update all arms with respective alpha dep. on chosen arm
        q = np.array([q[i] + ((alpha[index, i])*(r - q[index])) for i in range(len(q))])

        # add persevarative term to chosen arm
        chosen = np.zeros(arms)
        chosen[index] = 1
        h = h + sticky*(chosen - h)
        
        # if any q value is < 0 make it 0
        q[q<0] = 0
        q[q>1] = 1

        # check how many trials elapsed
        if trial == session_ends.iloc[sess_num]:
            sess_start=True
            sess_num += 1

    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

def load_data_into_dict(seed_n = 42):
    # # turn this into a function, call it to return sess_dict
    # sessdf = pd.read_csv('L:/4portProb_processed/sessdf.csv')
    # sessdf.drop(columns = 'Unnamed: 0', inplace = True)

    # exclude = ['[ 20  20  20 100]', '[0 0 0 0]', '[0]', '[0 0]',
    #     '[1000   80]', '[30]', '[40]', '[70]']
    # sessdf = sessdf[~sessdf.rewprobfull.isin(exclude)]
    # sessdf = sessdf[~sessdf.duplicated(subset = ['animal', 'session', 'trialstart', 'eptime'], keep = False)]

    # mask = (sessdf.task.isin(['unstr']))
    # sessdf_prep = data_prep(sessdf[mask], hist = 1, trialsinsess = 100, head= True)

    # # subselect for training/testing?
    # sessdf_prep['sess_bin'] = sessdf_prep.groupby(['animal', 'task'])['session'].transform(lambda x: pd.cut(x, bins=range(0, x.max() + 50, 50), labels=False, right=False)+1)
    # sessdf_prep = sessdf_prep[sessdf_prep.sess_bin>=4]
    # np.random.seed(seed_n)
    # test_sess_dict = {}
    # # pick 10% sessions for testing
    # for animal in sessdf_prep.animal.unique():
    #     unique_sess = sessdf_prep[sessdf_prep.animal == animal].session.nunique()
    #     test_sess = np.random.choice(sessdf_prep[sessdf_prep.animal == animal].session.unique(), int(unique_sess*0.2), replace = False)
    #     test_sess_dict[animal] = test_sess
    #     sessdf_prep.drop(sessdf_prep[(sessdf_prep.animal == animal) & (sessdf_prep.session.isin(test_sess))].index, inplace = True)

    # sess_dict = {key: group for key, group in sessdf_prep.groupby('animal')}

    # print('loaded data')
    # return sess_dict, test_sess_dict

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

from pybads import BADS
animals = ['test05022023','Blissey', 'Chikorita', 'Darkrai', 'Eevee',
            'Goldeen', 'Hoppip', 'Inkay', 'Jirachi',
            'Kirlia', 'Nidorina', 'Phione', 'Quilava', 'Raltz', 'Togepi',
            'Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian']
sess_dict, test_sess_dict = load_data_into_dict()
extra_params = 4
n_optim = 5
k = 7 # number of parameters for bic

options = {}
results = {}
filepath = 'L:/4portProb_simulations/model_parameters_20250826/alphamat_stickyscaled_first_block_bads'

tic()
for animal in animals:
    data = sess_dict[animal]
    n_trials = data.shape[0]
    
    def fun_for_pybads(x):
        return nllmatQlearning(x, data, extra_params)
    for n in range(n_optim):
        # run multiple optimizations
        print('Running optimization ' + str(n) + '...')
        options['random_seed'] = n
        bads = BADS(fun_for_pybads, None, [-1.0, -1.0, -1.0, -1.0, 0.0015, 0.0, 0.0], [1, 1, 1, 1, 100, 1.0, 10.0], options=options)
        # kept changing bounds so something would atleast work lol
        optimize_result = bads.optimize()
        results[animal] = [optimize_result.x, optimize_result.fval, optimize_result.success]
        nll = optimize_result.fval
        bic = k*np.log(n_trials) + 2*nll
         # open a file and store results
        with open(f'{filepath}.csv', 'a') as f:
            f.write(f'{animal},{n},{results[animal][0]},{results[animal][1]},{results[animal][2]},{test_sess_dict[animal]},{bic}\n')
with open(f'{filepath}.log', 'a') as f:
    f.write(toc())
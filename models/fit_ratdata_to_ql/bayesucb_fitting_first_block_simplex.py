import numpy as np
import datetime
from scipy.optimize import minimize
from utils.supplementaryFunctions import tic, toc

def nllBayesUCB(x0, df, arms):
    tau, c = x0
    alpha0, beta0 = 1, 1
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
        q = alphas/(alphas+betas)
        ucb = np.sqrt((alphas*betas)/(((alphas+betas)**2)*alphas+betas+np.ones(arms)))

        # softmax prob of choosing actions
        invtemp=1/tau
        P = np.exp(invtemp*(q+c*ucb))
        P = P/ np.sum(P) 

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
    with open('/mnt/pve/Homes/rishika/behavioralData/cleandf.pkl', 'rb') as f:
        df = pickle.load(f)

    # subset for expert performance
    subset_df = df[(df.sess_bin>=4) & (df.task == 'unstr')]
    from utils.dfLoading import subset_trials, add_block_groups
    
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

# fit using either diff-evo or scipy minimize
#animals = ['test05022023','Blissey', 'Chikorita', 'Darkrai', 'Eevee']
#animals = ['Goldeen', 'Hoppip', 'Inkay', 'Jirachi']
animals = ['Kirlia', 'Nidorina', 'Phione', 'Quilava', 'Raltz', 'Togepi']
#animals = ['Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian']
#animals = ['Emolga', 'Giratina', 'Haxorus', 'Ivysaur', 'Jigglypuff', 'Lugia']

train_data, test_data = load_data_into_dict()
extra_params = 4
n_optim = 3
k = 3 # number of parameters for bic
date = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
filepath = 'output/output_'+date
results = {}

# boundaries for parameters
bounds = ((0.0015, 100),    # tau = for softmax
          (0, 1))           # c = for ucb
 

tic()
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
for animal in animals:
    data = train_data[animal]
    n_trials = data.shape[0]

    for n in range(n_optim):
        # run multiple optimizations
        print('Running optimization ' + str(n) + '...')

        optimized_result = minimize(
                                    nllBayesUCB, 
                                    x0=[0.1, 0.1], 
                                    args = (data, extra_params),
                                    bounds = bounds,
                                    method = 'Nelder-Mead',
                                    options = {'disp':True}
                                    )

        results[animal] = [optimized_result.x, optimized_result.fun, optimized_result.success]
        nll = optimized_result.fun
        bic = k*np.log(n_trials) + 2*nll

        # open a file and store results
        with open(f'{filepath}.csv', 'a') as f:
            f.write(f'{animal},{n},{results[animal][0]},{results[animal][1]},{results[animal][2]},{test_data[animal]},{bic}\n')

# elapsed time
with open(f'{filepath}.log', 'a') as f:
    f.write(toc())
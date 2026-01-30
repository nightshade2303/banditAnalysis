import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier     # generate GP 
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel
from temp_utils.supplementaryFunctions import tic, toc
from scipy.special import expit
from scipy.optimize import minimize

def nLL_gp_ucb(x0, df, arms):
    beta, tau, ls, sigma_vert = x0
    ll = 0
    gamma = 1/tau
    beta_star = beta/tau
    sessions = df.session.value_counts()
    chosen_action = df.port.to_numpy()
    rewarded = df.reward.to_numpy()
    p = np.ones(len(df))*1e-6
    session_ends = np.cumsum(sessions)
    
    sess_num = 0
    sess_start = True

    # gp settings
    X = np.linspace(1,arms,arms).reshape(arms,1)
    knl = Product(RBF(length_scale=ls), ConstantKernel(constant_value=sigma_vert**2))

    for trial in range(len(chosen_action)):
        # check if sess restarted
        if sess_start == True:
            # if group.block_group.unique()[0] == 1:
            gp = GaussianProcessClassifier(kernel=knl, optimizer = None)
            start_trial = trial

        sess_start = False
        
        if len(np.unique(rewarded[start_trial:trial+1])) > 1:
            # what actions were taken so far
            a = chosen_action[start_trial:trial+1]

            # what rewards were given for each action
            r = rewarded[start_trial:trial+1]

            # update gp using all the info we got so far on actions and rew
            gp.fit(a.reshape(-1, 1), r)

            # get latent mean and variance
            mu, var = gp.latent_mean_and_variance(X)
            mu_star = expit(mu)
            sd_star = expit(mu + np.sqrt(var))

            # calculate probability of taking any action
            P = np.exp(gamma*mu_star + beta_star*sd_star)
            P = P/ np.sum(P)
            P = np.clip(P, a_min = 1e-6, a_max = None)
    
            # what was probability of action taken?
            chosen = int(chosen_action[trial]-1)
            p[trial] = P[chosen]

        # check how many trials elapsed
        if trial == session_ends.iloc[sess_num]:
            sess_start=True
            sess_num += 1

    ll += np.nansum(np.log(p))
    nll = -ll
    return nll

# load data - first block only and separate into test-train 20-80
def load_data_into_dict(seed_n = 42):
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


# fit using either diff-evo or scipy minimize
animals = ['test05022023','Blissey', 'Chikorita', 'Darkrai', 'Eevee',
            'Goldeen', 'Hoppip', 'Inkay', 'Jirachi',
            'Kirlia', 'Nidorina', 'Phione', 'Quilava', 'Raltz', 'Togepi',
            'Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian', 
            'Emolga', 'Giratina', 'Haxorus', 'Ivysaur', 'Jigglypuff', 'Lugia']

train_data, test_data = load_data_into_dict()
extra_params = 4
n_optim = 10
k = 4 # number of parameters for bic
filepath = 'L:/4portProb_simulations/model_parameters_20250919/gpc_first_block_bfgs'
results = {}

# boundaries for parameters
bounds = ((0, 10),      # beta = exploration coefficient
      (0.0015, 100),    # tau = for ucb softmax
      (0.001, 10),       # length scale of the RBF kernel; larger values = more spatial generalization
      (0.001, 10))       # vertical scaling of the RBF kernel 

tic()
for animal in animals:
    data = train_data[animal]
    n_trials = data.shape[0]

    for n in range(n_optim):
        # run multiple optimizations
        print('Running optimization ' + str(n) + '...')

        optimized_result = minimize(
                                    nLL_gp_ucb, 
                                    x0=[1, 1, 1, 1], 
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



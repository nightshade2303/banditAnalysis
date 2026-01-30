# load required packages
from opconNosepokeFunctions import *                              # load sessdf
from supplementaryFunctions import *                              # calc entropy etc.
from scipy.optimize import minimize                               # minimize log_lik for process_animal()

from sklearn.gaussian_process import GaussianProcessRegressor     # generate GP 
from sklearn.gaussian_process.kernels import RBF, DotProduct      # GP kernels
import time                                                       # spawn process with different seed each time
import multiprocessing as mp                                      # print current process when job taken

# load data first
sessdf = pd.read_csv('L:/4portProb_processed/sessdf.csv')
sessdf.drop(columns = 'Unnamed: 0', inplace = True)
window = 7
trialsinsess = 100
exclude = ['[ 20  20  20 100]', '[0 0 0 0]', '[0]', '[0 0]',
       '[1000   80]', '[30]', '[40]', '[70]']
sessdf = sessdf[~sessdf.rewprobfull.isin(exclude)]
sessdf = sessdf[~sessdf.duplicated(subset = ['animal', 'session', 'trialstart', 'eptime'], keep = False)]
sessdf = sessdf.groupby(['animal','session']).filter(lambda x: x.reward.size >= trialsinsess)

# set global variables and bounds for each parameter
N = 4           # number of arms 
T = 100         # number of trials to simulate

bounds = ((0.0001, 5),       # alpha = observation noise variance
          (0.0001, 10),      # beta = exploration coefficient
          (0.000001, 10),    # tau = for ucb softmax
          (0.001, 10),       # length scale of the RBF kernel; larger values = more spatial generalization
          (0.0001, 1))       # intial q value in a session

# fit all sessions on GP and get nLL
def nLL_gp_ucb(x0, sessdf, an):
    alpha, beta, tau, ls, q0 = x0
    
    gamma = 1/tau
    beta_star = beta/tau

    kernel = RBF(length_scale=ls)  #
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=None)
    X = np.linspace(1,N,N).reshape(N,1)
    
    for sess in sessdf.session.unique():
        session_data = sessdf[(sessdf.session == sess) & (sessdf.animal == an)]
        a = (session_data.port.astype(int)-1).to_numpy()
        r = session_data.reward.astype(int).to_numpy()
        m = np.full((T, N), q0)
        sd = np.full((T, N), np.sqrt(alpha))
        ll = np.zeros(T)
        nll = 0
        X_a = X[a]
        
        for t in range(T):
            # calculate probability of taking any action
            P = np.exp(gamma*m[t] + beta_star*sd[t])
            P = P/ np.sum(P)
    
            # what was probability of action taken?
            p = P[a[t]]
    
            # compute LL
            ll[t] = np.log(p)
    
            # what was action taken and reward received - fit gp to this and predict mu and sigma    
            if t < T-1:
                gp.fit(X_a[:(t+1)], r[:(t+1)])
                m[(t+1), :], sd[(t+1), :] = gp.predict(X, return_std=True)
        
            # compute nll
            nll += -np.nansum(ll)
        return nll

# run one process per animal to get somewhat faster results
def process_animal(an):
    # parameters
    params = np.zeros(len(bounds))

    # random seed generation for each process    
    # randomize parameters given bounds
    for i, bound in enumerate(bounds):
        (bound_low, bound_high) = bound
        np.random.seed(int(str(time.time_ns())[12:]))
        params[i] = np.random.uniform(bound_low, bound_high)
    
    # print params 
    print(f' initializing with --- {params}')

    # subselecting data (100 trials per sessiom and unstructured only rn)
    filtered = sessdf[(sessdf.animal == an) & (sessdf.task == 'unstr')].groupby('session').filter(lambda x: x.reward.size >= trialsinsess)
    filtered = filtered.groupby('session').head(trialsinsess)
    args = (filtered, an)
    
    print(mp.current_process())
    result = minimize(nLL_gp_ucb,
                      x0=params,
                      args=args,
                      method='Nelder-Mead',
                      bounds=bounds,
                      options={'maxiter': 1000, 'disp': True})
    return an, result.x, result.fun

# dummy process function
def my_func(x):
    time.sleep(.1)
    print(mp.current_process())
    return x * x
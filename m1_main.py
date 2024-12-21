import os
import pickle
import datetime
import numpy as np 
import pandas as pd
import multiprocessing as mp
from scipy.optimize import minimize
from scipy.special import erf
from scipy.stats import norm, uniform, gamma, halfnorm

# the path to the current file 
pth = os.path.dirname(os.path.abspath(__file__))

# constants
eps_, max_ = 1e-12, 1e12

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    y = np.exp(x)
    return y if y>1e-11 else 0

# -----------------------  Model ------------------------ #

class prospect_delegation:
    name = 'prospect_delegation'
    p_bnds   = [( 0, 1), (1,   4), (1e-3, 2), (0, 30), (-10, 10), (0, 10), (0, 10)]
    p_pbnds  = [(.1,.8), (1, 1.5), (.5, 1.5), (1, 5), (-1,  1), (1e-3,  1), (1e-3,  1)]
    p_names  = ['theta', 'lmbda', 'gamma', 'tau', 'b', 'kappa', 'sigma']  
    p_poi    = p_names
    p_priors = [uniform(0, 1), 
                uniform(0, 2), 
                uniform(0, 4), 
                gamma(2, 3), 
                norm(0, 2),
                halfnorm(0, 2), 
                halfnorm(0, 2)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: 1e-3 + 2/(1+clip_exp(-x)), 
                lambda x: 1e-3 + 4/(1+clip_exp(-x)), 
                lambda x: clip_exp(x), 
                lambda x: x,
                lambda x: clip_exp(x), 
                lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_)-np.log(1-x+eps_), 
                lambda x: np.log((x-1e-3))-np.log(2-(x-1e-3)), 
                lambda x: np.log((x-1e-3))-np.log(4-(x-1e-3)), 
                lambda x: np.log(x+eps_), 
                lambda x: x, 
                lambda x: np.log(x+eps_), 
                lambda x: np.log(x+eps_)]
    n_params = len(p_names)
    hidden_vars = ['v_gain', 'v_loss', 'u', 'p_fish', 'p_defer', 'p_safe', 'SubjectiveValue']

    def __init__(self, nA, params):
        self.nA = nA
        self.load_params(params)

    def load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.theta = params[0]
        self.alpha = .9#params[1]
        self.lmbda = params[1]
        self.gamma = params[2]
        self.tau   = params[3]
        self.b     = params[4]
        self.kappa = params[5]
        self.sigma = params[6]

    def policy(self, z_gain, z_loss, z_uncertain, x_gain=1, x_loss=-1):
        '''Make a choice

        z_gain: the gain for the current trial
        z_loss: the loss for the current trial
        z_uncertain: the uncertainty for the current trial
        '''
        # calculate the probability of gain & loss 
        eta = (z_gain*1 + z_loss*0 + z_uncertain*self.theta) / 10
        numer = eta**self.gamma
        denom = (eta**self.gamma + (1-eta)**self.gamma)**(1/self.gamma)
        pi_gain = numer/(denom+eps_)
        pi_loss = 1 - pi_gain

        # calculate the subjective value of gain & loss 
        self.v_gain = x_gain**self.alpha
        self.v_loss = -self.lmbda*(-x_loss)**self.alpha

        # calculate the utility
        self.u = pi_gain*self.v_gain + pi_loss*self.v_loss

        self.SubjectiveValue = self.u

        # calculate the decision policy 
        p_fish_tmp = 1/(1 + clip_exp(-self.tau*self.u))

        # calculate the defer rate
        m1 = (self.u/(self.sigma+eps_)/np.sqrt(2))+self.b+self.kappa
        m2 = (self.u/(self.sigma+eps_)/np.sqrt(2))+self.b-self.kappa
        self.p_defer = .5*(erf(m1) - erf(m2))

        # calculate the fishing/not fishing rate 
        self.p_fish = p_fish_tmp*(1-self.p_defer)
        self.p_safe = (1-self.p_defer)*(1-p_fish_tmp)
        pi = np.array([self.p_fish, self.p_defer, self.p_safe])
        pi /= pi.sum()
        return pi
    
# ----------------- Likelihood function ----------------- #

def loss_fn(params, sub_data, model_name, method='mle'):
    '''Total likelihood

        Fit individual:
            Maximum likelihood:
            log p(D|θ) = log \prod_i p(D_i|θ)
                       = \sum_i log p(D_i|θ )
            or Maximum a posterior 
            log p(θ|D) = \sum_i log p(D_i|θ ) + log p(θ)
    '''
    
    # instantiate the subject model
    nA = 3 # the number of possible choices
    model = eval(model_name)
    # instantiate the subject model for the block 
    subj = model(nA, params)
   
    # calculate the likelihood of the data
    ll = 0 
    for _, block_data in sub_data.items():
        
        for _, row in block_data.iterrows():
            # make a decision 
            z_gain = row['Zgain']
            z_loss = row['Zloss']
            z_uncertain = row['Uncertainty']
            response = row['Response']
            # if the participant response, calculate the likelihood
            if response: 
                pi = subj.policy(z_gain, z_loss, z_uncertain)
                a = row['Fishing']
                ll += np.log(pi[a]+eps_)
    loss = -ll

    # if method=='map', add log prior loss 
    if method=='map':
        lpr = 0
        for pri, fn, param in zip(model.p_priors, model.p_trans, params):
            lpr += np.max([pri.logpdf(fn(param)), -max_])
        loss += -lpr
    
    return loss 

# -------------------- Model Fitting -------------------- #

def fit(loss_fn, sub_data, model_name,
        bnds, pbnds, p_name, 
        method='mle', alg='Nelder-Mead', seed=2024, verbose=False):
    '''Fit the parameter using optimization 

    Args: 
        loss_fn: a function; log likelihood function
        sub_data: a dictionary, each key map a dataframe
        model_name: the name of the model 
        bnds: parameter bound
        pbnds: possible bound, used to initialize parameter
        p_name: the names of parameters
        method: decide if we use the prior -'mle', -'map'
        alg: the fiting algorithm, currently we can use 
            - 'Nelder-Mead': a simplex algorithm,
            - 'BFGS': a quasi-Newton algorithm, return hessian,
                        but only works on unconstraint problem
        seed:  random seed; used when doing parallel computing
        verbose: show the optimization details or not. 
    
    Return:
        result: optimization results
    '''
    # get some value
    n_params = len(p_name)
    # get the number of trial 
    n_rows = np.sum([sub_data[k].shape[0] for k in sub_data.keys()])

    # random init from the possible bounds 
    rng = np.random.RandomState(seed)
    param0 = [pbnd[0] + (pbnd[1] - pbnd[0]) * rng.rand() for pbnd in pbnds]
                    
    ## Fit the params 
    result = minimize(loss_fn, param0, args=(sub_data, model_name, method), 
                bounds=bnds, method=alg,
                options={'disp': verbose})
    x_min = result.x
            
    ## Save the optimize results 
    if verbose: 
        print(f'''  Fitted params: {x_min}, 
                    NLL: {result.fun}''')
    fit_res = {}
    fit_res['log_post']   = -result.fun
    fit_res['log_like']   = -loss_fn(x_min, sub_data, model_name, method)
    fit_res['param']      = x_min
    fit_res['param_name'] = p_name
    fit_res['n_param']    = n_params
    fit_res['aic']        = n_params*2 - 2*fit_res['log_like']
    fit_res['bic']        = n_params*np.log(n_rows) - 2*fit_res['log_like']
    
    return fit_res

def fit_parallel(data_set, model_name, method='mle', alg='Nelder-Mead', 
                 seed=2024, verbose=False, n_fits=40, n_cores=None):

    # load the data for fit
    fname = f'{pth}/data/{data_set}.pkl'
    with open(fname, 'rb') as handle: data = pickle.load(handle)

    # create the file to save fit results
    fname = f'{pth}/data/fit_info-{data_set}-{model_name}-{method}.pkl'
    # if file exists, load it, otherwise create it
    if os.path.exists(fname):
        # load the previous fit resutls
        with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
        fitted_sub_lst = [k for k in fit_sub_info.keys()]
    else:
        fitted_sub_lst = []
        fit_sub_info = {} 

    # initialize the pool for parallel computing 
    if n_cores is None: n_cores = int(mp.cpu_count() * .7)
    pool = mp.Pool(n_cores)

    # start fitting
    start_time = datetime.datetime.now()
    sub_start  = start_time

    # fit each subject
    done_subj = len(fitted_sub_lst)
    all_subj  = len(data.keys()) 

    # set up parameter bounds 
    p_bnds = eval(model_name).p_bnds
    p_pbnds = eval(model_name).p_pbnds
    # reparameterize if using BFGS
    if alg=='BFGS':
        p_bnds = None
        p_pbnds = [[fn(p) for p in pbnd] for fn, pbnd in 
                   zip(eval(model_name).p_links, eval(model_name).p_pbnds)]
    
    print('\nFitting the model...')
    for sub_id in data.keys():
        if sub_id not in fitted_sub_lst:  
            print(f'Fitting {model_name} subj {sub_id}, progress: {(done_subj*100)/all_subj:.2f}%')
            # put the loss into the computing pool
            results = [pool.apply_async(fit, args=(loss_fn, data[sub_id], model_name,
                                            p_bnds, 
                                            p_pbnds, 
                                            eval(model_name).p_names, 
                                            method, alg, seed+2*i, verbose))
                                            for i in range(n_fits)]
            # find the best fit result
            opt_val   = np.inf 
            losses, tol = [], 1e-3
            for p in results:
                res = p.get()
                losses.append(-res['log_post'])
                if -res['log_post'] < opt_val:
                    opt_val = -res['log_post']
                    opt_res = res
            n_low = (np.abs(np.array(losses)-opt_val)<tol).sum()
            print(f'\tNum of lowest loss: {n_low}/{len(losses)}')        
            # save the best fit result
            fit_sub_info[sub_id] = opt_res
            with open(fname, 'wb')as handle: pickle.dump(fit_sub_info, handle)
            sub_end = datetime.datetime.now()
            print(f'\tLOSS:{-opt_res["log_post"]:.4f}, using {(sub_end - sub_start).total_seconds():.2f} seconds')
            sub_start = sub_end
            done_subj += 1
    
    # END!!!
    end_time = datetime.datetime.now()
    print(f'\nparallel computing spend {(end_time - start_time).total_seconds():.2f} seconds')
    
# --------------  Latent variable inference -------------- #

def inference(data_set, model_name, method='mle'):

    # load the data for inference
    fname = f'{pth}/data/{data_set}.pkl'
    with open(fname, 'rb') as handle: data = pickle.load(handle)
    
    # load the fitted params
    fname = f'{pth}/data/fit_info-{data_set}-{model_name}-{method}.pkl'
    with open(fname, 'rb') as handle: fit_sub_info = pickle.load(handle)

    # infer the latent variables
    nA = 2 # the number of possible choices
    model = eval(model_name)

    infer_data = [] 
    print('\nInferring the latent variables...')
    for sub_id, sub_data in data.items():
        # get the fitted params for inference
        fitted_params = fit_sub_info[sub_id]['param']
        # fitted_params = [.5, 1, 1, 2, 1, 1, .1]
        # fitted_params = [f(p) for p, f in zip(fitted_params, model.p_links)]

        for sub_id in sub_data.keys():
            # instantiate the subject model
            block_data = sub_data[sub_id].copy()
            subj = model(nA, fitted_params)

            # initialize a blank dataframe to store the inference results
            col = ['ll'] + model.hidden_vars
            init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
            pred_data = pd.DataFrame(init_mat, columns=col)  

            for t, row in block_data.iterrows():
                # make a decision 
                z_gain = row['Zgain']
                z_loss = row['Zloss']
                z_uncertain = row['Uncertainty']
                response = row['Response']

                # if the participant response, calculate the likelihood
                if response: 
                    pi = subj.policy(z_gain, z_loss, z_uncertain)
                    a = row['Fishing']
                
                    # store the hidden variables 
                    pred_data.loc[t, 'll'] = np.log(pi[a]+eps_)
                    for v in model.hidden_vars: 
                        pred_data.loc[t, v] = eval(f'subj.{v}')
            
            # combine the subject data and inferred hidden variables
            pred_data = pred_data.dropna(axis=1, how='all')
            infer_datum = pd.concat([block_data, pred_data], axis=1)   
            infer_data.append(infer_datum)

    # combine the inferred data from all subjects and save it
    infer_data = pd.concat(infer_data, axis=0)
    infer_data.to_csv(f'{pth}/data/infer_data-{data_set}-{model_name}-{method}.csv', index=False)

if __name__ == '__main__':

    data_sets = ['leadership_data'] # 
    model_name = 'prospect_delegation'
    alg = 'BFGS'
    n_fits, n_cores = 100, 50

    for data_set in data_sets:

        # 1. fit the prospect delegation model
        fit_parallel(data_set, model_name, n_fits=n_fits, n_cores=n_cores, alg=alg)

        # 2. infer the latent variables using the prospect delegation model
        inference(data_set, model_name)


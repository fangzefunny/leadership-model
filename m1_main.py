import os
import pickle
import datetime
import numpy as np 
import pandas as pd
import multiprocessing as mp
from scipy.optimize import minimize

# the path to the current file 
pth = os.path.dirname(os.path.abspath(__file__))

# constants
eps_, max_ = 1e-12, 1e12

# -----------------------  Model ------------------------ #

class prospect_model:
    name = 'prospect_model'
    p_bnds   = [( 0, 1), (1, 50), ( 0, 1), (0, 50)]
    p_pbnds  = [(.1,.9), (1, 8),  (.1,.9),  (1, 8)]
    p_names  = ['alpha', 'lmbda', 'gamma', 'tau']  
    p_poi    = p_names
    p_priors = []
    p_trans  = []
    n_params = len(p_names)
    hidden_vars = ['v_gain', 'v_loss', 'u']

    def __init__(self, nA, params):
        self.nA = nA
        self.params(params)

    def load_params(self, params):
        self.theta = params[0]
        self.alpha = params[1]
        self.lmbda = params[2]
        self.gamma = params[3]
        self.tau   = params[4]

    def policy(self, x_gain, x_loss, z_gain, z_loss, z_amb):
        '''Make a choice

        x_gain: the gain for the current trial
        x_loss: the loss for the current trial
        eta: the probability of gain 
        '''
        # calculate the probability of gain & loss 
        eta = z_gain*1 + z_loss*0 + z_amb*self.theta
        numer = eta**self.gamma
        denom = (eta**self.gamma + (1-eta)**self.gamma)**(1/self.gamma)
        pi_gain = numer/denom
        pi_loss = 1 - pi_gain

        # calculate the subjective value of gain & loss 
        self.v_gain = x_gain**self.alpha
        self.v_loss = self.lmbda*(-x_loss)**self.alpha

        # calculate the utility
        self.u = pi_gain*self.v_gain - pi_loss*self.v_loss

        # calculate the decision policy 
        p_gain = 1/(1 + np.exp(-self.tau*self.u))
        p_loss = 1 - p_gain
        return np.array([p_loss, p_gain])
    
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
    nA = 2 # the number of possible choices
    model = eval(model_name)
   
    # calculate the likelihood of the data
    ll = 0 
    for _, block_data in sub_data.items():
        # instantiate the subject model for the block 
        subj = model(nA, params)

        for _, row in block_data.iterrows():
            # make a decision 
            x_gain = row['x_gain']
            x_loss = row['x_loss']
            z_gain = row['z_gain']
            z_loss = row['z_loss']
            z_amb  = row['z_amb']
            a      = row['a']
            pi = subj.policy(x_gain, x_loss, z_gain, z_loss, z_amb)
            ll += np.log(pi[a]+eps_)
    loss = -ll

    # if method=='map', add log prior loss 
    if method=='map':
        lpr = 0
        for pri, param in zip(model.p_priors, params):
            lpr += np.max([pri.logpdf(param), -max_])
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
    param0 = [pbnd[0] + (pbnd[1] - pbnd[0]
                ) * rng.rand() for pbnd in pbnds]
                    
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
    fit_res['log_post']   = result.fun
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
    fname = f'{pth}/data/fit_info-{model_name}-method.pkl'
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
    for sub_id in data.keys():
        if sub_id not in fitted_sub_lst:  
            print(f'Fitting {model_name} subj {sub_id}, progress: {(done_subj*100)/all_subj:.2f}%')
            # put the loss into the computing pool
            results = [pool.apply_async(loss_fn, args=(data[sub_id], model_name,
                                            eval(model_name).bnds, 
                                            eval(model_name).pbnds, 
                                            eval(model_name).p_name, 
                                            method, alg, seed+2*i, verbose))
                                            for i in range(n_fits)]
            # find the best fit result
            opt_val   = np.inf 
            losses, tol = [], 1e-2,
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

def inference(data_set, model_name):

    # load the data for inference
    fname = f'{pth}/data/{data_set}.pkl'
    with open(fname, 'rb') as handle: data = pickle.load(handle)
    
    # load the fitted params
    fname = f'{pth}/data/fit_info-{model_name}-method.pkl'
    with open(fname, 'rb') as handle: fit_sub_info = pickle.load(handle)

    # infer the latent variables
    nA = 2 # the number of possible choices
    model = eval(model_name)

    infer_data = [] 
    for sub_id, sub_data in data.items():
        # get the fitted params for inference
        fitted_params = fit_sub_info[sub_id]['param']
    
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
                x_gain = row['x_gain']
                x_loss = row['x_loss']
                z_gain = row['z_gain']
                z_loss = row['z_loss']
                z_amb  = row['z_amb']
                a      = row['a']
                pi = subj.policy(x_gain, x_loss, z_gain, z_loss, z_amb)
                
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
    infer_data.to_csv(f'{pth}/data/infer_data-{model_name}.csv', index=False)

if __name__ == '__main__':

    data_set = 'leadership_data'
    model_name = 'prospect_model'
    n_fits, n_cores = 40, 40

    # 1. fit the prospect model
    fit_parallel(data_set, model_name, n_fits=n_fits, n_cores=n_cores)

    # 2. infer the latent variables using the prospect model
    inference(data_set, model_name)


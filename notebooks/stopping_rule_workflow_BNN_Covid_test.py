import sys, os
import pickle
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
from scipy.stats import t
from itertools import product
import pystan

from viabel.vb import (mean_field_gaussian_variational_family,
                       mean_field_t_variational_family,
                       full_rank_gaussian_variational_family,
                       t_variational_family,
                       make_stan_log_density,
                       adagrad_optimize)


import  argparse
from posteriordb import PosteriorDatabase
import os

# please give path to your posteriordb installation here ......
pdb_path = os.path.join('/Users/alex/dev/posteriordb/', "posterior_database")
my_pdb = PosteriorDatabase(pdb_path)
pos = my_pdb.posterior_names()

modelcode = 2

from experiments import black_box_klvi, psis_correction
from viabel.functions import compute_posterior_moments
from data_generator import (data_generator_linear)

from viabel.vb import  rmsprop_IA_optimize_with_rhat, adam_IA_optimize_with_rhat
#from viabel.optimizers_avg_stopping_rule import  adam_IA_optimize_stop, adagrad_ia_optimize_stop, rmsprop_IA_optimize_stop
from viabel.optimizers_workflow import adagrad_workflow_optimize, rmsprop_workflow_optimize, adam_workflow_optimize

from viabel.data_process  import  Concrete
#approx= 'mf'

if modelcode == 1:
    # model_str = "ecdc0501-covid19imperial_v3"
    model_str = "mnist_100-nn_rbm1bJ10"
elif modelcode == 2:
    model_str = "ecdc0401-covid19imperial_v3"

posterior = my_pdb.posterior(model_str)
modelObject = posterior.model
data= posterior.data
code_string = modelObject.code('stan')
#text_file = open("stan_models/stan-covid19imperial_v2.stan", "w")

if modelcode == 1:
    try:
        # save the code string ....
        sm = pickle.load(open('stan_pkl/' + model_str + '.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=code_string)
        with open('stan_pkl/' + model_str + '.pkl', 'wb') as f:
            pickle.dump(sm, f)

    try:
        # save the posterior model .....
        sm, model_fit = pickle.load(open('stan_pkl/' + model_str + '_centered_posterior_samples.pkl', 'rb'))
    except:
        model_fit = sm.sampling(data=data.values(), iter=2000,
                                control=dict(adapt_delta=.96),
                                chains=4)
        with open('stan_pkl/' + model_str + '_posterior_samples.pkl', 'wb') as f:
            pickle.dump((sm, model_fit), f)

    approx = 'mf'

# code for covid19 model starts from here ...
elif modelcode == 2:
    try:
        # save the code string ....
        sm = pickle.load(open('stan_pkl/covid19_01_v3.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=code_string, model_name='covid19_model_2')
        with open('stan_pkl/covid19_01_v3.pkl', 'wb') as f:
            pickle.dump(sm, f)

    try:
        # save the posterior model .....
        model_fit = pickle.load(open('stan_pkl/covid19_v3_posterior_samples.pkl', 'rb'))[1]
    except:
        model_fit = sm.sampling(data=data.values(), iter=800,
                                control=dict(adapt_delta=.96), chains=1)
        with open('stan_pkl/covid19_v3_posterior_samples.pkl', 'wb') as f:
            pickle.dump((sm, model_fit), f)

    approx = 'fr'
#sm = pystan.StanModel(model_code=code_string, model_name='covid19_model_v2')
#model_fit = sm.sampling(data=data.values(), iter=600, chains=1)
K = len(model_fit.constrained_param_names())
print(K)
param_names =  model_fit.flatnames
# construct matrix of samples (both original and transformed) from non-centered model
samples_posterior = model_fit.to_dataframe(pars=model_fit.flatnames)
#samples_posterior['log_sigma'] = np.log(samples_posterior['sigma'])
samples_posterior = samples_posterior.loc[:,param_names].values.T

true_mean = np.mean(samples_posterior, axis=1)
true_cov = np.cov(samples_posterior)
true_sigma = np.sqrt(np.diag(np.atleast_1d(true_cov)))
log_density = make_stan_log_density(model_fit)

true_mean_pmz = true_mean[:K]
true_sigma_pmz = true_sigma[:K]

mf_gaussian = mean_field_gaussian_variational_family(K)
fr_gaussian = t_variational_family(K, df=10000000)
init_param_fr = np.concatenate([np.zeros(K), np.ones(int(K*(K+1)/2))])
#init_param_mf = np.concatenate([np.zeros(K), np.ones(K)])
init_param_mf = np.concatenate([true_mean_pmz, np.log(true_sigma_pmz)])

klvi_fr_objective_and_grad = black_box_klvi(fr_gaussian, log_density, 100)
klvi_mf_objective_and_grad = black_box_klvi(mf_gaussian, log_density, 100)

# define the relevant functions based on the approximate density.
if approx == 'mf':
    fn_density = mf_gaussian
    init_var_param = init_param_mf
    obj_and_grad = klvi_mf_objective_and_grad
else:
    fn_density = fr_gaussian
    init_var_param = init_param_fr
    obj_and_grad = klvi_fr_objective_and_grad

print('########### HMC Mean #################')
print(true_mean)
print(true_sigma)
n_samples=100000

stopping_rule = 1
stopping_rule_str = "elbo" if stopping_rule == 1 else "mcse"
a, b, c, d, e = \
    adagrad_workflow_optimize(10000, obj_and_grad, init_var_param,
                              K, learning_rate=.0004, n_optimizers=1, tolerance=0.05,
                              stopping_rule=stopping_rule)
samples, smoothed_log_weights, khat = psis_correction(log_density, fn_density,
                                                      b[-1], n_samples)
samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(log_density, fn_density,
                                                               a, n_samples)

print(true_mean)
print(b[-1][:K])
print('khat:', khat)
print('khat ia:', khat_ia)
cov_iters_fr_rms = fn_density.mean_and_cov(b[-1])[1]
cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(a)[1]
print('Difference between analytical mean and HMC mean:',
      np.sqrt(np.mean(np.square(b[-1][:K].flatten() - true_mean[:K].flatten()))))
print('Difference between analytical cov and HMC cov:',
      np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov[:K,:K].flatten()))))

print('Difference between analytical mean and HMC mean-IA:',
        np.sqrt(np.mean(np.square(a[:K].flatten() - true_mean.flatten()))))
print('Difference between analytical cov and HMC cov-IA:',
        np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))

with open('stan_pkl/' + model_str + '_' + stopping_rule_str + '_chain.pkl', 'wb') as f:
    pickle.dump((a, b), f)

rmse = np.sqrt(np.mean(np.square(b[i][:K].flatten() - true_mean.flatten())))


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
pdb_path = os.path.join('/Users/akashd/Desktop/research_repos/posteriordb/posteriordb/', "posterior_database")
my_pdb = PosteriorDatabase(pdb_path)
pos = my_pdb.posterior_names()



parser = argparse.ArgumentParser()
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--optimizer', default='rmsprop', type=str)
args = parser.parse_args()



from experiments import black_box_klvi, psis_correction
from viabel.functions import compute_posterior_moments
from data_generator import (data_generator_linear)

from viabel.vb import  rmsprop_IA_optimize_with_rhat, adam_IA_optimize_with_rhat
#from viabel.optimizers_avg_stopping_rule import  adam_IA_optimize_stop, adagrad_ia_optimize_stop, rmsprop_IA_optimize_stop
from viabel.optimizers_workflow import adagrad_workflow_optimize, rmsprop_workflow_optimize, adam_workflow_optimize


from viabel.data_process  import   Concrete
model1 = 'eight-school'
model2 = 'radon'
model3 = 'concrete'
model4 = 'linear-sim'
model5 = 'robust-reg'
model6= 'radon2'
model7 = 'radon-full'
model8= 'bnn1'
model9 ='bnn2'
model10 = 'covid19-1'
model11 = 'covid19-2'
model12 = 'covid19-3'
model13 = 'covid19-4'
model = 14

approx = 'fr'
#approx= 'mf'


if args.model == 1:
    model = model1
elif args.model ==2:
    model= model2
elif args.model ==3:
    model= model3
elif args.model==4:
    model= model4
elif args.model==5:
    model = model5
elif args.model ==6:
    model = model6
elif args.model ==7:
    model = model7
    posterior = my_pdb.posterior("radon_all-radon_county_intercept")
elif args.model == 8:
    model = model8
    #posterior = my_pdb.posterior('')
elif args.model == 9:
    model = model9
elif args.model ==10:
    model = model10
    posterior = my_pdb.posterior("ecdc0501-covid19imperial_v2")
    modelObject = posterior.model
    data= posterior.data
    code_string = modelObject.code('stan')
    #text_file = open("stan_models/stan-covid19imperial_v2.stan", "w")
    with open('stan_models/stan-covid19imperial_v2.stan', 'w') as stan_file:
        stan_file.write(code_string)


elif args.model == 11:
    model = model11
    posterior = my_pdb.posterior("ecdc0501-covid19imperial_v3")
    modelObject = posterior.model
    data= posterior.data
    code_string = modelObject.code('stan')
    #text_file = open("stan_models/stan-covid19imperial_v2.stan", "w")
    with open('stan_models/stan-covid19imperial_v3.stan', 'w') as stan_file:
        stan_file.write(code_string)

elif args.model ==12:
    model = model12
    posterior = my_pdb.posterior("ecdc0401-covid19imperial_v2")
    modelObject = posterior.model
    data= posterior.data
    code_string = modelObject.code('stan')
    #text_file = open("stan_models/stan-covid19imperial_v2.stan", "w")
    with open('stan_models/stan-covid19-0401-imperial_v2.stan', 'w') as stan_file:
        stan_file.write(code_string)

elif args.model == 13:
    model = model13
    posterior = my_pdb.posterior("ecdc0401-covid19imperial_v3")
    modelObject = posterior.model
    data= posterior.data
    code_string = modelObject.code('stan')
    #text_file = open("stan_models/stan-covid19imperial_v2.stan", "w")
    with open('stan_models/stan-covid19-0401-imperial_v3.stan', 'w') as stan_file:
        stan_file.write(code_string)



if args.optimizer in ["adagrad", "adam", "rmsprop"]:
    optimizer = args.optimizer
else:
    optimizer = "adagrad"


def tranform_to_theta(ncp_samples):
    ncp_samples_tranformed = ncp_samples.copy()
    ncp_samples_tranformed[2:] = (ncp_samples_tranformed[0]
                                  + np.exp(ncp_samples_tranformed[1]) * ncp_samples_tranformed[2:])
    return ncp_samples_tranformed


def get_ncp_approx_samples(var_family, opt_param, n_samples):
    ncp_samples = var_family.sample(opt_param, n_samples).T
    return ncp_samples, tranform_to_theta(ncp_samples)



if model == model4:
    regression_model_code = """data {
      int<lower=0> N;   // number of observations
      int<lower=0> D;   // number of observations
      matrix[N, D] x;   // predictor matrix
      vector[N] y;      // outcome vector
    }
    parameters {
      vector[D] beta;       // coefficients for predictors
    }

    model {
      beta ~ normal(0, 2.);
      #y ~ student_t(10000000, x * beta, 1);  // likelihood
      y ~ normal( x * beta, 0.5);  // likelihood
    }"""

    try:
        sm = pickle.load(open('linear_reg_model_41.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=regression_model_code, model_name='regression_model')
        with open('linear_reg_model_41.pkl', 'wb') as f:
            pickle.dump(sm, f)


    N = 120
    k = 30
    #SEED = 5080
    SEED=210
    alpha = 1.
    noise_sigma = 0.5
    noise_var = noise_sigma ** 2
    rho = 0.6
    regression_data = data_generator_linear(N, k, alpha=alpha,
                                            noise_variance=noise_var,
                                            rho=rho, seed=SEED)
    X = regression_data['X']
    Y = regression_data['Y']
    Y = Y[:, 0]
    W = regression_data['W']

    optimizer = 'rmsprop'
    data = dict(N=N, x=X, y=Y, D=k)
    fit = sm.sampling(data=data)

    prior_mean = np.zeros((k, 1))
    prior_covariance = np.eye(k)
    true_mean, true_cov = compute_posterior_moments(prior_mean,
                                                    prior_covariance,
                                                    noise_var, X, Y[:, None])

    true_mean = true_mean.flatten()

    true_std = np.sqrt(np.diag(true_cov))

    # print(posterior_mean)
    # print('Difference between analytical mean and HMC mean:', np.sum(np.square(posterior_mean_hmc - true_mean)))
    # print('Difference between analytical covariance and HMC covariance:', np.sum(np.square(posterior_cov_hmc - true_cov)))
    # print('Difference between analytical std dev and HMC std dev:', np.sum(np.square(posterior_std_hmc - true_std)))
    # print('Difference between analytical z-score and HMC z-score:', np.sqrt(np.mean(np.square((posterior_mean_hmc - true_mean)/true_std))))

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    fr_g_var_family = t_variational_family(k, df=1000000)

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    stan_log_density = make_stan_log_density(fit)
    klvi_mf_objective_and_grad = black_box_klvi(mf_g_var_family, stan_log_density, 100)
    klvi_fr_objective_and_grad = black_box_klvi(fr_g_var_family, stan_log_density, 100)
    init_mean = np.zeros(k)
    init_log_std = np.ones(k)
    init_var_param = np.concatenate([init_mean, init_log_std])
    n_iters = 5000

    klvi_mf_objective_and_grad_pd = black_box_klvi(mf_g_var_family, stan_log_density, 100)
    klvi_fr_objective_and_grad_pd = black_box_klvi(fr_g_var_family, stan_log_density, 100)
    init_mean = np.zeros(k)
    # init_mean_random = np.random.normal([k], stddev=1.0)
    init_log_std = np.ones(k)
    # init_log_std_random = tf.random.normal([k], stddev=1.)
    init_var_param = np.concatenate([init_mean, init_log_std])
    init_fr_var_param = np.concatenate([init_mean, np.ones(int(k * (k + 1) / 2))])

    if approx == 'mf':
        fn_density = mf_g_var_family
        init_var_param = init_var_param
        obj_and_grad = klvi_mf_objective_and_grad
    else:
        fn_density = fr_g_var_family
        init_var_param = init_fr_var_param
        obj_and_grad = klvi_fr_objective_and_grad


    if optimizer == 'rmsprop':
        klvi_var_param_rms, klvi_var_param_list_rms, avg_klvi_mean_list_rms, avg_klvi_sigmas_list_rms, klvi_history_rms, _, op_log_mf_rms = \
            rmsprop_workflow_optimize(11000, obj_and_grad, init_var_param, k, learning_rate=.012, n_optimisers=1, stopping_rule=2, tolerance=0.02, plotting=True)

        n_samples = 20000
        ia_var_params=  np.concatenate((avg_klvi_mean_list_rms[0], avg_klvi_sigmas_list_rms[0]), axis=0)
        print(ia_var_params)

        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                             klvi_var_param_list_rms[0,-1,:], n_samples)

        samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(stan_log_density, fn_density,
                                                             ia_var_params, n_samples)

        print(true_mean)
        print(klvi_var_param_rms[:k])
        print('khat:', khat)
        print('khat-ia:', khat_ia)
        true_mean, true_cov = compute_posterior_moments(prior_mean,
                                                        prior_covariance,
                                                        noise_var, X, Y[:, None])

        cov_iters_fr_rms = fr_g_var_family.mean_and_cov(klvi_var_param_rms)[1]
        cov_iters_fr_rms_ia = fr_g_var_family.mean_and_cov(ia_var_params)[1]
        print('Difference between analytical mean and HMC mean:', np.sqrt(np.mean(np.square(klvi_var_param_rms[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

        print('Difference between analytical mean and HMC mean-IA:', np.sqrt(np.mean(np.square(ia_var_params[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia.flatten() - true_cov.flatten()))))


    elif optimizer == 'adam':
        klvi_var_param_adam, klvi_var_param_list_adam, avg_klvi_mean_list_adam, avg_klvi_sigmas_list_adam, klvi_history_adam, _, op_log_mf_adam = \
            adam_workflow_optimize(11000, obj_and_grad, init_var_param, k, learning_rate=.01, n_optimisers=1,stopping_rule=2, tolerance=0.02, plotting=True)

        n_samples = 40000
        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                             klvi_var_param_adam, n_samples)
        print(true_mean)
        print(klvi_var_param_adam[:k])
        print('khat:', khat)
        true_mean, true_cov = compute_posterior_moments(prior_mean,
                                                        prior_covariance,
                                                        noise_var, X, Y[:, None])

        cov_iters_fr_rms = [fr_g_var_family.mean_and_cov(x)[1] for x in [klvi_var_param_adam]]
        print('Difference between analytical mean and HMC mean:', np.sqrt(np.mean(np.square(klvi_var_param_adam[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms[-1].flatten() - true_cov.flatten()))))

    elif optimizer == 'adagrad':
        a, b, c, d, e = adagrad_workflow_optimize(12000, obj_and_grad, init_fr_var_param, k, learning_rate=0.01, tolerance=0.01, n_optimizers=1,stopping_rule=2,
                                                  tail_avg_iters=200, plotting=False)

        n_samples = 40000
        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                              b[-1], n_samples)

        samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(stan_log_density, fn_density,
                                                              a, n_samples)
        print(true_mean)
        print(b[-1][:k])
        print('khat:', khat)
        print('khat ia:', khat_ia)
        true_mean, true_cov = compute_posterior_moments(prior_mean,
                                                        prior_covariance,
                                                        noise_var, X, Y[:, None])

        cov_iters_fr_rms = [fr_g_var_family.mean_and_cov(x)[1] for x in b]
        cov_iters_fr_rms_ia1 = fr_g_var_family.mean_and_cov(a)[1]
        print('Difference between analytical mean and HMC mean:', np.sqrt(np.mean(np.square(b[-1][:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms[-1].flatten() - true_cov.flatten()))))

        print('Difference between analytical mean and HMC mean-IA:', np.sqrt(np.mean(np.square(a[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov-IA:', np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))



######   model for robust regression ....
elif model == model5:

    regression_model_code = """data {
      int<lower=0> N;   // number of observations
      matrix[N, 2] x;   // predictor matrix
      vector[N] y;      // outcome vector
      real<lower=1> df; // degrees of freedom
    }
    parameters {
      vector[2] beta;       // coefficients for predictors
    }

    model {
      beta ~ normal(0, 10);
      y ~ student_t(df, x * beta, 1);  // likelihood
    }"""


    try:
        sm = pickle.load(open('robust_reg_model_41.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=regression_model_code, model_name='regression_model')
        with open('robust_reg_model_41.pkl', 'wb') as f:
            pickle.dump(sm, f)

    np.random.seed(5039)
    beta_gen = np.array([-2, 1])
    N = 25
    x = np.random.randn(N, 2).dot(np.array([[1, .75], [.75, 1]]))
    y_raw = x.dot(beta_gen) + np.random.standard_t(40, N)
    y = y_raw - np.mean(y_raw)

    data = dict(N=N, x=x, y=y, df=40)
    fit = sm.sampling(data=data, iter=50000, thin=50, chains=10)

    true_mean = np.mean(fit['beta'], axis=0)
    true_cov = np.cov(fit['beta'].T)
    print('true mean =', true_mean)
    print('true cov =', true_cov)

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    fr_g_var_family = t_variational_family(k, df=1000000)

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    stan_log_density = make_stan_log_density(fit)
    klvi_mf_objective_and_grad = black_box_klvi(mf_g_var_family, stan_log_density, 100)
    klvi_fr_objective_and_grad = black_box_klvi(fr_g_var_family, stan_log_density, 100)
    init_mean = np.zeros(k)
    init_log_std = np.ones(k)
    init_var_param = np.concatenate([init_mean, init_log_std])
    n_iters = 5000

    klvi_mf_objective_and_grad_pd = black_box_klvi(mf_g_var_family, stan_log_density, 100)
    klvi_fr_objective_and_grad_pd = black_box_klvi(fr_g_var_family, stan_log_density, 100)
    init_mean = np.zeros(k)
    # init_mean_random = np.random.normal([k], stddev=1.0)
    init_log_std = np.ones(k)
    # init_log_std_random = tf.random.normal([k], stddev=1.)
    init_var_param = np.concatenate([init_mean, init_log_std])
    init_fr_var_param = np.concatenate([init_mean, np.ones(int(k * (k + 1) / 2))])

    if approx == 'mf':
        fn_density = mf_g_var_family
        init_var_param = init_var_param
        obj_and_grad = klvi_mf_objective_and_grad
    else:
        fn_density = fr_g_var_family
        init_var_param = init_fr_var_param
        obj_and_grad = klvi_fr_objective_and_grad


elif model == model3:
    CData = Concrete()
    X1, Y1 = CData.get_normalised_data()

    print(X1.shape)
    print(Y1.shape)
    # exit()
    N_train, k = X1.shape

    linear_reg_code = """
    data{
        int<lower=0> N;
        int<lower=0> K;
        matrix[N,K] X;
        vector[N] y;
        real<lower=0> sigma;
    }
    parameters{
    vector[K] w;
    #real<lower=0> sigma;
    }
    model{
    w ~ normal(0,1);
    y ~ normal(X*w , sigma);
    }
    generated quantities{
    real log_density;
    #log_density = normal_lpdf(y|X*w, sigma) + normal_lpdf(w| 0, 1) + gamma_lpdf(sigma|0.5, 0.5) + log(sigma);
    #log_density = normal_lpdf(y|X*w, sigma) + normal_lpdf(w| 0, 1)+ gamma_lpdf(sigma|0.5, 0.5) + log(sigma);
    log_density = normal_lpdf(y|X*w, sigma) + normal_lpdf(w| 0, 1);
    }
    """

    model_data = {'N': N_train,
                  'K': k,
                  'y': Y1[:, 0],
                  'X': X1,
                  'sigma':0.2

                  }

    try:
        sm = pickle.load(open('linear_reg_chains_concrete5.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=linear_reg_code)
        with open('linear_reg_chains_concrete5.pkl', 'wb') as f:
            pickle.dump(sm, f)

    fit_hmc = sm.sampling(data=model_data, iter=2400)

    la = fit_hmc.extract(permuted=True)
    hmc_w = la['w']
    #hmc_residual = la['sigma']
    # stan_sigma = la['sigma']
    # stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta
    params_hmc = hmc_w
    # params_stan = np.concatenate((stan_w, stan_sigma), axis=0)
    true_mean = np.mean(params_hmc, axis=0)
    true_cov = np.cov(params_hmc.T)
    params_hmc_sq = np.mean(params_hmc ** 2, axis=0)
    params_hmc_sigmas = np.std(params_hmc, axis=0)

    #params_hmc_residual_mean = np.mean(hmc_residual, axis=0)
    #params_hmc_residual_sq = np.mean(hmc_residual ** 2, axis=0)
    #params_hmc_residual_sigmas = np.std(hmc_residual, axis=0)
    print('##### HMC Mean####')

    num_proposal_samples = 50000

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    fr_g_var_family = t_variational_family(k, df=1000000)

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    stan_log_density = make_stan_log_density(fit_hmc)
    klvi_mf_objective_and_grad = black_box_klvi(mf_g_var_family, stan_log_density, 100)
    klvi_fr_objective_and_grad = black_box_klvi(fr_g_var_family, stan_log_density, 100)
    init_mean = np.zeros(k)
    init_log_std = np.ones(k)
    init_var_param = np.concatenate([init_mean, init_log_std])
    n_iters = 9000

    klvi_mf_objective_and_grad_pd = black_box_klvi(mf_g_var_family, stan_log_density, 100)
    klvi_fr_objective_and_grad_pd = black_box_klvi(fr_g_var_family, stan_log_density, 100)
    init_mean = np.zeros(k)
    # init_mean_random = np.random.normal([k], stddev=1.0)
    init_log_std = np.ones(k)
    # init_log_std_random = tf.random.normal([k], stddev=1.)
    init_var_param = np.concatenate([init_mean, init_log_std])
    init_fr_var_param = np.concatenate([init_mean, np.ones(int(k * (k + 1) / 2))])
    optimizer = 'rmsprop'

    if approx == 'mf':
        fn_density = mf_g_var_family
        init_var_param = init_var_param
        obj_and_grad = klvi_mf_objective_and_grad
    else:
        fn_density = fr_g_var_family
        init_var_param = init_fr_var_param
        obj_and_grad = klvi_fr_objective_and_grad

    if optimizer == 'rmsprop':
        klvi_var_param_rms, klvi_var_param_list_rms, avg_klvi_mean_list_rms, avg_klvi_sigmas_list_rms, klvi_history_rms, _, op_log_mf_rms = \
            rmsprop_workflow_optimize(11000, obj_and_grad, init_var_param, k, learning_rate=.006, n_optimisers=1, stopping_rule=1, tolerance=0.005)

        n_samples = 20000
        ia_var_params=  np.concatenate((avg_klvi_mean_list_rms[0], avg_klvi_sigmas_list_rms[0]), axis=0)
        print(ia_var_params)

        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                             klvi_var_param_list_rms[0,-1,:], n_samples)

        samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(stan_log_density, fn_density,
                                                             ia_var_params, n_samples)

        print(true_mean)
        print(klvi_var_param_rms[:k])
        print('khat:', khat)
        print('khat-ia:', khat_ia)

        cov_iters_fr_rms = fr_g_var_family.mean_and_cov(klvi_var_param_rms)[1]
        cov_iters_fr_rms_ia = fr_g_var_family.mean_and_cov(ia_var_params)[1]
        print('Difference between analytical mean and HMC mean:', np.sqrt(np.mean(np.square(klvi_var_param_rms[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

        print('Difference between analytical mean and HMC mean-IA:', np.sqrt(np.mean(np.square(ia_var_params[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia.flatten() - true_cov.flatten()))))


    elif optimizer == 'adam':
        klvi_var_param_adam, klvi_var_param_list_adam, avg_klvi_mean_list_adam, avg_klvi_sigmas_list_adam, klvi_history_adam, _, op_log_mf_adam = \
            adam_workflow_optimize(11000, obj_and_grad, init_var_param, k, learning_rate=.02,n_optimisers=1)


        n_samples = 40000
        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                             klvi_var_param_adam, n_samples)

        print(true_mean)
        print(klvi_var_param_adam[:k])
        print('khat:', khat)

        cov_iters_fr_rms = [fr_g_var_family.mean_and_cov(x)[1] for x in [klvi_var_param_adam]]
        print('Difference between analytical mean and HMC mean:', np.sqrt(np.mean(np.square(klvi_var_param_adam[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms[-1].flatten() - true_cov.flatten()))))

    elif optimizer == 'adagrad':
        a, b, c, d, e = adagrad_workflow_optimize(8000, obj_and_grad, init_fr_var_param, k, learning_rate=0.005, tolerance=0.005,
                                                  n_optimizers=1, stopping_rule=2)
        n_samples = 40000
        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                              b[-1], n_samples)

        samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(stan_log_density, fn_density,
                                                              a, n_samples)
        print(true_mean)
        print(b[-1][:k])
        print('khat:', khat)
        print('khat ia:', khat_ia)

        cov_iters_fr_rms = fr_g_var_family.mean_and_cov(b[-1])[1]
        cov_iters_fr_rms_ia1 = fr_g_var_family.mean_and_cov(a)[1]

        print('Difference between analytical mean and HMC mean:', np.sqrt(np.mean(np.square(b[-1][:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

        print('Difference between analytical mean and HMC mean-IA:', np.sqrt(np.mean(np.square(a[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov-IA:', np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))


elif model == model1:
    J = 8
    y = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
    sigma = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
    data = dict(J=J, y=y, sigma=sigma)

    pmz = 'cp'
    optimiser = 'rmsprop'
    try:
        cp = pickle.load(open('stan_pkl/eight_schools_cp.pkl', 'rb'))
    except:
        cp = pystan.StanModel(file='stan_models/eight_schools_cp.stan', model_name='eight_School_cp_model')
        with open('stan_pkl/eight_schools_cp.pkl', 'wb') as f:
            pickle.dump(cp, f)

    try:
        ncp = pickle.load(open('stan_pkl/eight_schools_ncp.pkl', 'rb'))
    except:
        ncp = pystan.StanModel(file='eight_schools_ncp.stan', model_name='eight_School_ncp_model')
        with open('stan_pkl/eight_schools_ncp.pkl', 'wb') as f:
            pickle.dump(ncp, f)



    try:
        eight_schools_cp_fit = pickle.load(open('stan_pkl/eight_schools_cp_posterior_samples.pkl', 'rb'))
    except:
        eight_schools_cp_fit = cp.sampling(data=data, iter=11000, warmup=1000,control=dict(adapt_delta=.99), chains=1)
        with open('stan_pkl/eight_schools_cp_posterior_samples.pkl', 'wb') as f:
            pickle.dump(eight_schools_cp_fit, f)

    try:
        eight_schools_ncp_fit = pickle.load(open('stan_pkl/eight_schools_ncp_posterior_samples.pkl', 'rb'))
    except:
        eight_schools_ncp_fit = ncp.sampling(data=data, iter=32000, warmup=2000, thin=3,
                                                 control=dict(adapt_delta=.95), chains=1)
        with open('stan_pkl/eight_schools_ncp_posterior_samples.pkl', 'wb') as f:
            pickle.dump(eight_schools_ncp_fit, f)


    # number of parameters and parameter names in centered model
    n_params_cp = len(eight_schools_cp_fit.constrained_param_names())
    param_names_cp = ['mu', 'log_tau'] + eight_schools_cp_fit.flatnames[2:n_params_cp]

    # number of parameters and parameter names in non-centered model
    n_params_ncp = len(eight_schools_ncp_fit.constrained_param_names())
    param_names_ncp = ['mu', 'log_tau'] + eight_schools_ncp_fit.flatnames[2:n_params_ncp]
    param_names_ncp_transformed = ['mu', 'log_tau'] + eight_schools_ncp_fit.flatnames[n_params_ncp:]

    # construct matrix of samples (both original and transformed) from non-centered model
    samples_ncp_df = eight_schools_ncp_fit.to_dataframe(pars=eight_schools_ncp_fit.flatnames)
    samples_ncp_df['log_tau'] = np.log(samples_ncp_df['tau'])
    samples_ncp = samples_ncp_df.loc[:, param_names_ncp].values.T
    samples_ncp_transformed = samples_ncp_df.loc[:, param_names_ncp_transformed].values.T

    # use samples from non-centered model for ground true mean and covariance
    true_mean_ncp = np.mean(samples_ncp, axis=1)
    true_cov_ncp = np.cov(samples_ncp)
    true_sigma_ncp = np.sqrt(np.diag(true_cov_ncp))
    true_mean_ncp_tranformed = np.mean(samples_ncp_transformed, axis=1)
    true_cov_ncp_tranformed = np.cov(samples_ncp_transformed)

    true_mean_cp = true_mean_ncp.copy()
    true_mean_cp[2:] = true_mean_cp[0] + true_mean_ncp[2:] * np.exp(true_mean_cp[1])

    k = 10
    eight_schools_cp_log_density = make_stan_log_density(eight_schools_cp_fit)
    eight_schools_ncp_log_density = make_stan_log_density(eight_schools_ncp_fit)
    mf_gaussian_cp = mean_field_gaussian_variational_family(n_params_cp)
    mf_gaussian_ncp = mean_field_gaussian_variational_family(n_params_ncp)
    fr_gaussian_cp = t_variational_family(n_params_cp, df=1000000)
    fr_gaussian_ncp = t_variational_family(n_params_ncp, df=1000000)

    init_param1 = np.concatenate([np.zeros(k), np.ones(k)])
    init_fr_param1 = np.zeros(int(k * (k + 3) / 2))

    klvi_mf_objective_and_grad_cp = black_box_klvi(mf_gaussian_cp, eight_schools_cp_log_density, 100)
    klvi_mf_objective_and_grad_ncp = black_box_klvi(mf_gaussian_cp, eight_schools_ncp_log_density, 100)
    klvi_fr_objective_and_grad_cp = black_box_klvi(fr_gaussian_cp, eight_schools_cp_log_density, 100)
    klvi_fr_objective_and_grad_ncp = black_box_klvi(fr_gaussian_ncp, eight_schools_ncp_log_density, 100)
    print('eight-school model:')

    n_samples = 40000
    true_cov = true_cov_ncp_tranformed

    if pmz == 'ncp':
        stan_log_density = eight_schools_ncp_log_density
        true_mean = true_mean_ncp
        true_cov_ncp = true_cov_ncp_tranformed
        fn_density = fr_gaussian_ncp
    elif pmz == 'cp':
        stan_log_density = eight_schools_cp_log_density
        true_mean = true_mean_cp
        fn_density = fr_gaussian_cp


    if optimiser == 'rmsprop':
        if pmz == 'ncp':
            klvi_fr_var_param_rms_ncp1, klvi_fr_var_param_list_rms_ncp1, avg_klvi_fr_mean_list_rms_ncp1, avg_klvi_fr_sigmas_list_rms_ncp1, klvi_fr_history_rms_ncp1, _, op_log_fr_rms_ncp1 = \
                rmsprop_workflow_optimize(8500, klvi_fr_objective_and_grad_ncp, init_fr_param1, k, learning_rate=.008, tolerance=0.03, n_optimisers=1, stopping_rule=2)
        elif pmz == 'cp':
            klvi_fr_var_param_rms_ncp1, klvi_fr_var_param_list_rms_ncp1, avg_klvi_fr_mean_list_rms_ncp1, avg_klvi_fr_sigmas_list_rms_ncp1, klvi_fr_history_rms_ncp1, _, op_log_fr_rms_ncp1 = \
                rmsprop_workflow_optimize(10000, klvi_fr_objective_and_grad_cp, init_fr_param1, k, learning_rate=.009, n_optimisers=1, tolerance=0.008, stopping_rule=2, tail_avg_iters=300, plotting=True)


        ia_var_params=  np.concatenate((avg_klvi_fr_mean_list_rms_ncp1[0], avg_klvi_fr_sigmas_list_rms_ncp1[0]), axis=0)
        print(ia_var_params)
        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                              klvi_fr_var_param_list_rms_ncp1[0,-1,:], n_samples)
        samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(stan_log_density, fn_density,
                                                                       ia_var_params, n_samples)

        print(true_mean)
        print(klvi_fr_var_param_list_rms_ncp1[0,-1,:k])
        print('khat:', khat)
        print('khat ia:', khat_ia)
        cov_iters_fr_rms = fn_density.mean_and_cov(klvi_fr_var_param_rms_ncp1)[1]
        cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(ia_var_params)[1]
        print('Difference between analytical mean and HMC mean:',
              np.sqrt(np.mean(np.square(klvi_fr_var_param_rms_ncp1[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:',
              np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))
        print('Difference between analytical mean and HMC mean-IA:',
              np.sqrt(np.mean(np.square(ia_var_params[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov-IA:',
              np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))

    elif optimiser == 'adagrad':
        if pmz == 'ncp':
            a, b, c, d, e = \
                adagrad_workflow_optimize(8500, klvi_fr_objective_and_grad_ncp, init_fr_param1, k,  learning_rate=.008, tolerance=0.020, stopping_rule=1)
        elif pmz == 'cp':
            a, b,c,d,e  = \
                adagrad_workflow_optimize(10000, klvi_fr_objective_and_grad_cp, init_fr_param1,k, learning_rate=.009, tolerance=0.020, stopping_rule=2)


        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                              b[-1], n_samples)
        samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(stan_log_density, fn_density,
                                                                       a, n_samples)
        print(true_mean)
        print(b[-1][:k])
        print('khat:', khat)
        print('khat ia:', khat_ia)

        cov_iters_fr_rms = fn_density.mean_and_cov(b[-1])[1]
        cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(a)[1]
        print('Difference between analytical mean and HMC mean:',
              np.sqrt(np.mean(np.square(b[-1][:k].flatten() - true_mean.flatten ()))))
        print('Difference between analytical cov and HMC cov:',
              np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

        print('Difference between analytical mean and HMC mean-IA:',
              np.sqrt(np.mean(np.square(a[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov-IA:',
              np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))

elif model == model6:
    import pandas as pd

    optimiser = 'adagrad'
    n_samples= 20000
    sub_model = 'unpooled'
    try:
        unp = pickle.load(open('radon_unpooled.pkl', 'rb'))
    except:
        unp = pystan.StanModel(file='radon_unpooled_model.stan', model_name='radon_unpooled')
        with open('radon_unpooled.pkl', 'wb') as f:
            pickle.dump(unp, f)

    srrs2 = pd.read_csv('radon/srrs2.dat')
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2.assign(fips=srrs2.stfips*1000 + srrs2.cntyfips)[srrs2.state=='MN']
    srrs_all = srrs2.assign(fips=srrs2.stfips*1000 + srrs2.cntyfips)[srrs2.state=='']

    cty = pd.read_csv('radon/cty.dat')
    cty_mn = cty[cty.st=='MN'].copy()
    cty_mn[ 'fips'] = 1000*cty_mn.stfips + cty_mn.ctfips
    cty_all = cty[cty.st=='*'].copy()
    cty_all[ 'fips'] = 1000*cty_all.stfips + cty_all.ctfips

    srrs_all = srrs_all.merge(cty_all[['fips', 'Uppm']], on='fips')
    srrs_all = srrs_all.drop_duplicates(subset='idnum')
    u = np.log(srrs_all.Uppm)

    N = len(srrs_all)

    srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
    srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
    u = np.log(srrs_mn.Uppm)

    n = len(srrs_mn)

    srrs_mn.county = srrs_mn.county.str.strip()
    mn_counties = srrs_mn.county.unique()
    counties = len(mn_counties)


    county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
    county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
    radon = srrs_mn.activity
    srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
    floor_measure = srrs_mn.floor.values
    unpooled_data = {'N': len(log_radon),
                   'county': county+1,
                   'x': floor_measure,
                   'y': log_radon}

    unpooled_fit = unp.sampling(data=unpooled_data, iter=2000, chains=2)

    n_params_unp = len(unpooled_fit.constrained_param_names())

    param_names_unp = ['beta', 'log_sigma'] + unpooled_fit.flatnames[2:n_params_unp]
    # construct matrix of samples (both original and transformed) from non-centered model
    samples_unp = unpooled_fit.to_dataframe(pars=unpooled_fit.flatnames)
    samples_unp['log_sigma'] = np.log(samples_unp['sigma'])
    samples_unp = samples_unp.loc[:,param_names_unp].values.T
    # use samples from non-centered model for ground true mean and covariance
    true_mean_unp = np.mean(samples_unp, axis=1)
    true_cov_unp = np.cov(samples_unp)
    true_sigma_unp = np.sqrt(np.diag(true_cov_unp))
    radon_unp_log_density = make_stan_log_density(unpooled_fit)

    true_mean = true_mean_unp
    true_cov = true_cov_unp
    k=n_params_unp
    n_params_unp = len(unpooled_fit.constrained_param_names())
    mf_gaussian_unp = mean_field_gaussian_variational_family(n_params_unp)
    fr_gaussian_unp = t_variational_family(n_params_unp, df=10000)

    # initialisation ...
    init_param_unp_fr = np.concatenate([np.zeros(n_params_unp), np.ones(int(n_params_unp*(n_params_unp+1)/2))])
    init_param_unp_mf = np.concatenate([np.zeros(n_params_unp), np.ones(n_params_unp)])

    klvi_fr_objective_and_grad_unp = black_box_klvi(fr_gaussian_unp, radon_unp_log_density, 100)
    approx= 'fr'


    if approx == 'mf':
        fn_density = mf_gaussian_unp
        init_var_param = init_param_unp_mf
        obj_and_grad = klvi_mf_objective_and_grad_unp
    else:
        fn_density = fr_gaussian_unp
        init_var_param = init_param_unp_fr
        obj_and_grad = klvi_fr_objective_and_grad_unp


    if sub_model == 'unpooled':
        if optimiser == 'rmsprop':
            klvi_var_param_rms_unp, klvi_var_param_list_rms_unp, avg_klvi_mean_list_rms_unp, avg_klvi_sigmas_list_rms_unp, klvi_history_rms_unp, _, op_log_fr_rms_unp = \
                rmsprop_workflow_optimize(14000, klvi_fr_objective_and_grad_unp, init_var_param,
                                          n_params_unp, learning_rate=.013,n_optimisers=1, tolerance=0.02, tail_avg_iters=200, stopping_rule=1)

            ia_var_params=  np.concatenate((avg_klvi_mean_list_rms_unp[0], avg_klvi_sigmas_list_rms_unp[0]), axis=0)
            print(ia_var_params)
            samples, smoothed_log_weights, khat = psis_correction(radon_unp_log_density, fn_density,
                                                                  klvi_var_param_list_rms_unp[0,-1,:], n_samples)
            samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(radon_unp_log_density, fn_density,
                                                                           ia_var_params, n_samples)

            print(true_mean)
            print(klvi_var_param_list_rms_unp[0,-1,:k])
            print('khat:', khat)
            print('khat ia:', khat_ia)
            cov_iters_fr_rms = fn_density.mean_and_cov(klvi_var_param_rms_unp)[1]
            cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(ia_var_params)[1]
            print('Difference between analytical mean and HMC mean:',
                  np.sqrt(np.mean(np.square(klvi_var_param_rms_unp[:k].flatten() - true_mean.flatten()))))
            print('Difference between analytical cov and HMC cov:',
                  np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))
            print('Difference between analytical mean and HMC mean-IA:',
                  np.sqrt(np.mean(np.square(ia_var_params[:k].flatten() - true_mean.flatten()))))
            print('Difference between analytical cov and HMC cov-IA:',
                  np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))

        elif optimiser == 'adagrad':
            a,b,c,d,e = \
                adagrad_workflow_optimize(10000, klvi_fr_objective_and_grad_unp, init_var_param,
                                          n_params_unp, learning_rate=.010,n_optimizers=1, tolerance=0.02, stopping_rule=2)
            samples, smoothed_log_weights, khat = psis_correction(radon_unp_log_density, fn_density,
                                                                  b[-1], n_samples)
            samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(radon_unp_log_density, fn_density,
                                                                           a, n_samples)
            print(true_mean)
            print(b[-1][:k])
            print('khat:', khat)
            print('khat ia:', khat_ia)
            cov_iters_fr_rms = fn_density.mean_and_cov(b[-1])[1]
            cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(a)[1]
            print('Difference between analytical mean and HMC mean:',
                  np.sqrt(np.mean(np.square(b[-1][:k].flatten() - true_mean.flatten ()))))
            print('Difference between analytical cov and HMC cov:',
                  np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

            print('Difference between analytical mean and HMC mean-IA:',
                  np.sqrt(np.mean(np.square(a[:k].flatten() - true_mean.flatten()))))
            print('Difference between analytical cov and HMC cov-IA:',
                  np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))

elif model == model10:
    print('lol')
    #print(data.values())
    #exit()

    try:
        sm = pickle.load(open('covid19_01_v2.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=code_string, model_name='covid19_model')
        with open('covid19_01_v2.pkl', 'wb') as f:
            pickle.dump(sm, f)


    try:
        model_fit = pickle.load(open('stan_pkl/covid19_posterior_samples.pkl', 'rb'))
    except:
        model_fit = sm.sampling(data=data.values(), iter=800,
                                                 control=dict(adapt_delta=.96), chains=1)
        with open('stan_pkl/covid19_posterior_samples.pkl', 'wb') as f:
            pickle.dump(model_fit, f)

    #print(model_fit)
    #exit()
    #sm = pystan.StanModel(model_code=code_string, model_name='covid19_model_v2')
    #model_fit = sm.sampling(data=data.values(), iter=600, chains=1)
    K = len(model_fit.constrained_param_names())
    print(K)
    param_names =  model_fit.flatnames
    # construct matrix of samples (both original and transformed) from non-centered model
    samples_posterior = model_fit.to_dataframe(pars=model_fit.flatnames)
    #samples_posterior['log_sigma'] = np.log(samples_posterior['sigma'])
    samples_posterior = samples_posterior.loc[:,param_names].values.T

    print(samples_posterior.shape)
    true_mean = np.mean(samples_posterior, axis=1)
    true_cov = np.cov(samples_posterior)
    true_sigma = np.sqrt(np.diag(true_cov))
    covid19_log_density = make_stan_log_density(model_fit)

    true_mean_pmz = true_mean[:K]
    true_sigma_pmz = true_sigma[:K]

    mf_gaussian = mean_field_gaussian_variational_family(K)
    fr_gaussian = t_variational_family(K, df=10000000)
    print(true_mean.shape)

    init_param_fr = np.concatenate([np.zeros(K), np.ones(int(K*(K+1)/2))])
    init_param_mf = np.concatenate([np.ones(K), np.ones(K)])
    init_param_mf = np.concatenate([true_mean_pmz, np.log(true_sigma_pmz)])

    klvi_fr_objective_and_grad = black_box_klvi(fr_gaussian, covid19_log_density, 100)
    klvi_mf_objective_and_grad = black_box_klvi(mf_gaussian, covid19_log_density, 10000)
    approx= 'mf'
    if approx == 'mf':
        fn_density = mf_gaussian
        init_var_param = init_param_mf
        obj_and_grad = klvi_mf_objective_and_grad
    else:
        fn_density = fr_gaussian
        init_var_param = init_param_fr
        obj_and_grad = klvi_fr_objective_and_grad

    print(true_sigma)
    n_samples=100000

    a, b, c, d, e = \
        adagrad_workflow_optimize(10000, obj_and_grad, init_var_param,
                                  K, learning_rate=.0000, n_optimizers=1, tolerance=0.05, stopping_rule=1)
    samples, smoothed_log_weights, khat = psis_correction(covid19_log_density, fn_density,
                                                          b[-1], n_samples)
    samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(covid19_log_density, fn_density,
                                                                   a, n_samples)
    print(true_mean)
    print(b[-1][:K])
    print('khat:', khat)
    print('khat ia:', khat_ia)
    cov_iters_fr_rms = fn_density.mean_and_cov(b[-1])[1]
    cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(a)[1]
    print('Difference between analytical mean and HMC mean:',
          np.sqrt(np.mean(np.square(b[-1][:K].flatten() - true_mean.flatten()))))
    print('Difference between analytical cov and HMC cov:',
          np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

    print('Difference between analytical mean and HMC mean-IA:',
          np.sqrt(np.mean(np.square(a[:K].flatten() - true_mean.flatten()))))
    print('Difference between analytical cov and HMC cov-IA:',
          np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))


elif model == model11:
    print('lol')
    #print(data.values())
    #exit()

    try:
        sm = pickle.load(open('stan_pkl/covid19_01_v3.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=code_string, model_name='covid19_model_2')
        with open('stan_pkl/covid19_01_v3.pkl', 'wb') as f:
            pickle.dump(sm, f)


    try:
        model_fit = pickle.load(open('stan_pkl/covid19_v3_posterior_samples.pkl', 'rb'))
    except:
        model_fit = sm.sampling(data=data.values(), iter=800,
                                                 control=dict(adapt_delta=.96), chains=1)
        with open('stan_pkl/covid19_v3_posterior_samples.pkl', 'wb') as f:
            pickle.dump(model_fit, f)

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
    true_sigma = np.sqrt(np.diag(true_cov))
    covid19_log_density = make_stan_log_density(model_fit)

    true_mean_pmz = true_mean[:K]
    true_sigma_pmz = true_sigma[:K]

    mf_gaussian = mean_field_gaussian_variational_family(K)
    fr_gaussian = t_variational_family(K, df=10000000)

    init_param_fr = np.concatenate([np.zeros(K), np.ones(int(K*(K+1)/2))])
    #init_param_mf = np.concatenate([np.zeros(K), np.ones(K)])
    init_param_mf = np.concatenate([true_mean_pmz, np.log(true_sigma_pmz)])

    klvi_fr_objective_and_grad = black_box_klvi(fr_gaussian, covid19_log_density, 100)
    klvi_mf_objective_and_grad = black_box_klvi(mf_gaussian, covid19_log_density, 100)
    approx= 'mf'
    if approx == 'mf':
        fn_density = mf_gaussian
        init_var_param = init_param_mf
        obj_and_grad = klvi_mf_objective_and_grad
    else:
        fn_density = fr_gaussian
        init_var_param = init_param_fr
        obj_and_grad = klvi_fr_objective_and_grad

    print(true_mean)
    print(true_sigma)
    n_samples=100000

    a, b, c, d, e = \
        adagrad_workflow_optimize(10000, obj_and_grad, init_var_param,
                                  K, learning_rate=.0004, n_optimizers=1, tolerance=0.05, stopping_rule=1)
    samples, smoothed_log_weights, khat = psis_correction(covid19_log_density, fn_density,
                                                          b[-1], n_samples)
    samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(covid19_log_density, fn_density,
                                                                   a, n_samples)
    print(true_mean)
    print(b[-1][:K])
    print('khat:', khat)
    print('khat ia:', khat_ia)
    cov_iters_fr_rms = fn_density.mean_and_cov(b[-1])[1]
    cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(a)[1]
    print('Difference between analytical mean and HMC mean:',
          np.sqrt(np.mean(np.square(b[-1][:K].flatten() - true_mean.flatten()))))
    print('Difference between analytical cov and HMC cov:',
          np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

    print('Difference between analytical mean and HMC mean-IA:',
          np.sqrt(np.mean(np.square(a[:K].flatten() - true_mean.flatten()))))
    print('Difference between analytical cov and HMC cov-IA:',
          np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))


elif model == model12:
    print('lol')
    #print(data.values())
    #exit()

    try:
        sm = pickle.load(open('stan_pkl/covid19_04_v2.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=code_string, model_name='covid19_model_3')
        with open('stan_pkl/covid19_04_v2.pkl', 'wb') as f:
            pickle.dump(sm, f)


    try:
        model_fit = pickle.load(open('stan_pkl/covid19_posterior_samples_04_01.pkl', 'rb'))
    except:
        model_fit = sm.sampling(data=data.values(), iter=800,
                                                 control=dict(adapt_delta=.96), chains=1)
        with open('stan_pkl/covid19_posterior_samples_04_01.pkl', 'wb') as f:
            pickle.dump(model_fit, f)

    #print(model_fit)
    #exit()
    #sm = pystan.StanModel(model_code=code_string, model_name='covid19_model_v2')
    #model_fit = sm.sampling(data=data.values(), iter=600, chains=1)
    K = len(model_fit.constrained_param_names())
    print(K)
    param_names =  model_fit.flatnames
    # construct matrix of samples (both original and transformed) from non-centered model
    samples_posterior = model_fit.to_dataframe(pars=model_fit.flatnames)
    #samples_posterior['log_sigma'] = np.log(samples_posterior['sigma'])
    samples_posterior = samples_posterior.loc[:,param_names].values.T

    print(samples_posterior.shape)
    true_mean = np.mean(samples_posterior, axis=1)
    true_cov = np.cov(samples_posterior)
    true_sigma = np.sqrt(np.diag(true_cov))
    covid19_log_density = make_stan_log_density(model_fit)

    true_mean_pmz = true_mean[:K]
    true_sigma_pmz = true_sigma[:K]

    mf_gaussian = mean_field_gaussian_variational_family(K)
    fr_gaussian = t_variational_family(K, df=10000000)
    print(true_mean.shape)

    init_param_fr = np.concatenate([np.zeros(K), np.ones(int(K*(K+1)/2))])
    init_param_mf = np.concatenate([np.ones(K), np.ones(K)])
    init_param_mf = np.concatenate([true_mean_pmz, np.log(true_sigma_pmz)])

    klvi_fr_objective_and_grad = black_box_klvi(fr_gaussian, covid19_log_density, 100)
    klvi_mf_objective_and_grad = black_box_klvi(mf_gaussian, covid19_log_density, 10000)
    approx= 'mf'
    if approx == 'mf':
        fn_density = mf_gaussian
        init_var_param = init_param_mf
        obj_and_grad = klvi_mf_objective_and_grad
    else:
        fn_density = fr_gaussian
        init_var_param = init_param_fr
        obj_and_grad = klvi_fr_objective_and_grad

    print(true_sigma)
    n_samples=100000

    a, b, c, d, e = \
        adagrad_workflow_optimize(10000, obj_and_grad, init_var_param,
                                  K, learning_rate=.0000, n_optimizers=1, tolerance=0.05, stopping_rule=1)
    samples, smoothed_log_weights, khat = psis_correction(covid19_log_density, fn_density,
                                                          b[-1], n_samples)
    samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(covid19_log_density, fn_density,
                                                                   a, n_samples)
    print(true_mean)
    print(b[-1][:K])
    print('khat:', khat)
    print('khat ia:', khat_ia)
    cov_iters_fr_rms = fn_density.mean_and_cov(b[-1])[1]
    cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(a)[1]
    print('Difference between analytical mean and HMC mean:',
          np.sqrt(np.mean(np.square(b[-1][:K].flatten() - true_mean.flatten()))))
    print('Difference between analytical cov and HMC cov:',
          np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

    print('Difference between analytical mean and HMC mean-IA:',
          np.sqrt(np.mean(np.square(a[:K].flatten() - true_mean.flatten()))))
    print('Difference between analytical cov and HMC cov-IA:',
          np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))





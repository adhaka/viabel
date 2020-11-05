#!/usr/bin/env python3

import sys, os
sys.path.append('..')
sys.path.append('../..')
import autograd
import pickle
import pystan

import seaborn as sns
import autograd.numpy as np

from viabel import all_bounds
from viabel.vb import (mean_field_t_variational_family,
                       t_variational_family,
                       black_box_klvi,
                       black_box_chivi,
                       make_stan_log_density,
                       make_stan_log_density_grad,
                       adagrad_optimize,
                       markov_score_climbing_cis)

from experiments import (get_samples_and_log_weights,
                         improve_with_psis,
                         plot_history,
                         plot_approx_and_exact_contours,
                         plot_dist_to_opt_param,
                         check_approx_accuracy,
                         print_bounds)

from viabel.vb_overdispersed import (black_box_gapis,
                                     adagrad_optimize_IS)

from data_generator import (data_generator_linear)

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
    sm = pickle.load(open('robust_reg_model_inc.pkl', 'rb'))
except:
    sm = pystan.StanModel(model_code=regression_model_code, model_name='regression_model')
    with open('robust_reg_model_inc.pkl', 'wb') as f:
        pickle.dump(sm, f)

np.random.seed(5039)
beta_gen = np.array([-2, 1])
N = 25
x = np.random.randn(N, 2).dot(np.array([[1,.75],[.75, 1]]))
y_raw = x.dot(beta_gen) + np.random.standard_t(40, N)
y = y_raw - np.mean(y_raw)

data = dict(N=N, x=x, y=y, df=40)
fit = sm.sampling(data=data, iter=50000, thin=50, chains=10)

true_mean = np.mean(fit['beta'], axis=0)
true_cov = np.cov(fit['beta'].T)

os.makedirs('../figures', exist_ok=True)
lims = dict(xlim=[-4,-1], ylim=[-.5,3.5])
mc_samples = 1000000  # number of Monte Carlo samples for estimating bounds and for PSIS

mf_t_var_family = mean_field_t_variational_family(2, 40)
stan_log_density = make_stan_log_density(fit)
klvi_objective_and_grad = black_box_klvi(mf_t_var_family, stan_log_density, 100)

init_mean    = np.zeros(2)
init_log_std = np.ones(2)
init_var_param = np.concatenate([init_mean, init_log_std])
n_iters = 5000

klvi_var_param, klvi_param_history, _, klvi_history, _ = \
    adagrad_optimize(n_iters, klvi_objective_and_grad, init_var_param, learning_rate=.01)

check_approx_accuracy(mf_t_var_family, klvi_var_param, true_mean, true_cov, verbose=True);

k = 2
mf_t_var_family= mean_field_t_variational_family(k, df=40)
stan_log_density = make_stan_log_density(fit)
stan_log_density_grad= make_stan_log_density_grad(fit)

incl_klvi_mf_objective_and_grad = markov_score_climbing_cis(mf_t_var_family, stan_log_density, 2000,2)

inc_klvi_var_param, inc_klvi_param_history, _,  inc_klvi_history, op_log_inklvi = \
    adagrad_optimize(n_iters, incl_klvi_mf_objective_and_grad, init_var_param, learning_rate=.005, has_log_norm=3, k=2)


mf_t_var_family = mean_field_t_variational_family(k, 10)
stan_log_density = make_stan_log_density(fit)

init_mean    = np.zeros(k)
init_log_std = np.ones(k)*0.5
init_var_param = np.concatenate([init_mean, init_log_std])
n_iters = 3000

check_approx_accuracy(mf_t_var_family, inc_klvi_var_param, true_mean, true_cov, verbose=True);

mf_t_var_family= mean_field_t_variational_family(k, df=40)
chivi_mf_objective_and_grad_pd = black_box_chivi(2, mf_t_var_family, stan_log_density, 10000)
chivi_var_param, chivi_param_history, _,  chivi_history, op_log_chivi = \
    adagrad_optimize(n_iters, chivi_mf_objective_and_grad, init_var_param, learning_rate=.005, has_log_norm=3, k=2)

check_approx_accuracy(mf_t_var_family, chivi_var_param, true_mean, true_cov, verbose=True);

from collections import namedtuple

from autograd import value_and_grad, vector_jacobian_product, jacobian, elementwise_grad
from autograd.extend import primitive, defvjp

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist
from autograd.scipy.linalg import sqrtm
from scipy.linalg import eigvalsh
from  .optimization_diagnostics import autocorrelation, monte_carlo_se, monte_carlo_se2, compute_khat_iterates, gpdfit
from .psis import psislw

from paragami import (PatternDict,
                      NumericVectorPattern,
                      PSDSymmetricMatrixPattern,
                      FlattenFunctionInput,
                      NumericArrayPattern)

from functools import partial

from autograd.core import getval
from autograd import grad

import tqdm
import scipy.stats as stats
from .tempering_schedule import  adaptive_tempering, sigmoid_tempering, telescope_tempering

from ._distributions import multivariate_t_logpdf

from .functions import compute_R_hat, compute_R_hat_adaptive_numpy, compute_R_hat_halfway, stochastic_iterate_averaging
from .functions import flat_to_triang, triang_to_flat, dlogq_dmu_t, dlogq_dsigma_t


__all__ = [
    'mean_field_gaussian_variational_family',
    'mean_field_t_variational_family',
    't_variational_family',
    'black_box_klvi',
    'black_box_klvi_pd',
    'black_box_klvi_pd2',
    'black_box_chivi',
    'markov_score_climbing_cis',
    'make_stan_log_density',
    'adagrad_optimize',
    'rmsprop_IA_optimize_with_rhat',
    'adam_IA_optimize_with_rhat'
]

VariationalFamily = namedtuple('VariationalFamily',
                               ['sample', 'entropy',
                                'logdensity', 'mean_and_cov',
                                'pth_moment', 'var_param_dim'])



def mean_field_gaussian_variational_family(dim):
    rs = npr.RandomState(0)
    def unpack_params(var_param):
        mean, log_std = var_param[:dim], var_param[dim:]
        return mean, log_std

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        mean, log_std = unpack_params(var_param)
        return my_rs.randn(n_samples, dim) * np.exp(log_std) + mean

    def entropy(var_param):
        mean, log_std = unpack_params(var_param)
        return 0.5 * dim * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    def logdensity(x, var_param):
        mean, log_std = unpack_params(var_param)
        return mvn.logpdf(x, mean, np.diag(np.exp(2*log_std)))

    def mean_and_cov(var_param):
        mean, log_std = unpack_params(var_param)
        return mean, np.diag(np.exp(2*log_std))

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        _, log_std = unpack_params(var_param)
        vars = np.exp(2*log_std)
        if p == 2:
            return np.sum(vars)
        else:  # p == 4
            return 2*np.sum(vars**2) + np.sum(vars)**2

    return VariationalFamily(sample, entropy, logdensity,
                             mean_and_cov, pth_moment, 2*dim)




def full_rank_gaussian_variational_family(dim):
    rs = npr.RandomState(0)

    def beta_to_L(beta):
        print(beta.shape)
        L = flat_to_triang(beta)
        L= L[0]
        return L

    def L_to_beta(L):
        return triang_to_flat(L)

    def unpack_params(var_param):
        mean, beta = var_param[:dim], var_param[dim:]
        return mean, beta

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        mean, beta = unpack_params(var_param)
        L = beta_to_L(beta)
        return np.dot( my_rs.randn(n_samples, dim), L) + mean
        #return my_rs.randn(n_samples, dim) @ L + mean

    def entropy(var_param):
        mean, beta = unpack_params(var_param)
        L = beta_to_L(beta[:,np.newaxis])
        return np.sum(np.log(np.diag(L))) + 0.5*dim* (1 + np.log(2 * np.pi))
        #return 0.5 * dim * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    def logdensity(x, var_param):
        mean, beta = unpack_params(var_param)
        L = beta_to_L(beta[:,np.newaxis])
        Sigma = L@L.T
        return mvn.logpdf(x, mean, Sigma)

    def mean_and_cov(var_param):
        mean, beta = unpack_params(var_param)
        L =  beta_to_L(beta[:,np.newaxis])
        Sigma = L@L.T
        return mean, np.diag(Sigma)

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        _, log_std = unpack_params(var_param)
        vars = np.exp(2*log_std)
        if p == 2:
            return np.sum(vars)
        else:  # p == 4
            return 2*np.sum(vars**2) + np.sum(vars)**2


    return VariationalFamily(sample, entropy, logdensity, mean_and_cov, pth_moment, dim*(dim+3)//2)


def mean_field_t_variational_family(dim, df):
    if df <= 2:
        raise ValueError('df must be greater than 2')
    rs = npr.RandomState(0)
    def unpack_params(var_param):
        mean, log_scale = var_param[:dim], var_param[dim:]
        return mean, log_scale

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        mean, log_scale = unpack_params(var_param)
        return mean + np.exp(log_scale)*my_rs.standard_t(df, size=(n_samples, dim))

    def entropy(var_param):
        # ignore terms that depend only on df
        mean, log_scale = unpack_params(var_param)
        return np.sum(log_scale)

    def logdensity(x, var_param):
        mean, log_scale = unpack_params(var_param)
        if x.ndim == 1:
            x = x[np.newaxis,:]
        return np.sum(t_dist.logpdf(x, df, mean, np.exp(log_scale)), axis=-1)

    def mean_and_cov(var_param):
        mean, log_scale = unpack_params(var_param)
        return mean, df / (df - 2) * np.diag(np.exp(2*log_scale))

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        if df <= p:
            raise ValueError('df must be greater than p')
        _, log_scale = unpack_params(var_param)
        scales = np.exp(log_scale)
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(scales**2)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(scales**4) + np.sum(scales**2)**2)

    return VariationalFamily(sample, entropy, logdensity,
                             mean_and_cov, pth_moment, 2*dim)


def _get_mu_sigma_pattern(dim):
    ms_pattern = PatternDict(free_default=True)
    ms_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_pattern['Sigma'] = PSDSymmetricMatrixPattern(size=dim)
    return ms_pattern


def _get_low_rank_mu_sigma_pattern(dim,M):
    ms_lr_pattern = PatternDict(free_default=True)
    ms_lr_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_lr_pattern['B'] =  NumericArrayPattern(shape=(dim,M))
    ms_lr_pattern['D'] = NumericArrayPattern(shape=(dim,))
    #ms_pattern['Lr'] =
    return ms_lr_pattern



def t_variational_family(dim, df):
    if df <= 2:
        raise ValueError('df must be greater than 2')
    rs = npr.RandomState(0)
    ms_pattern = _get_mu_sigma_pattern(dim)

    logdensity = FlattenFunctionInput(
        lambda x, ms_dict: multivariate_t_logpdf(x, ms_dict['mu'], ms_dict['Sigma'], df),
        patterns=ms_pattern, free=True, argnums=1)

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        s = np.sqrt(my_rs.chisquare(df, n_samples) / df)
        param_dict = ms_pattern.fold(var_param)
        z = my_rs.randn(n_samples, dim)
        sqrtSigma = sqrtm(param_dict['Sigma'])
        return param_dict['mu'] + np.dot(z, sqrtSigma)/s[:,np.newaxis]

    def entropy(var_param):
        # ignore terms that depend only on df
        param_dict = ms_pattern.fold(var_param)
        return .5*np.log(np.linalg.det(param_dict['Sigma']))

    def mean_and_cov(var_param):
        param_dict = ms_pattern.fold(var_param)
        return param_dict['mu'], df / (df - 2.) * param_dict['Sigma']

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = ms_pattern.fold(var_param)
        sq_scales = np.linalg.eigvalsh(param_dict['Sigma'])
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(sq_scales)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(sq_scales**2) + np.sum(sq_scales)**2)

    return VariationalFamily(sample, entropy, logdensity, mean_and_cov,
                             pth_moment, ms_pattern.flat_length(True))



def low_rank_gaussian_variational_family(dim, M, df=1e14):
    if df <= 2:
        raise ValueError('df must be greater than 2')
    rs = npr.RandomState(0)
    ms_lr_pattern = _get_low_rank_mu_sigma_pattern(dim, M)

    logdensity = FlattenFunctionInput(
        lambda x, ms_lr_dict: multivariate_t_logpdf(x, ms_lr_dict['mu'], np.dot(ms_lr_dict['B'], np.transpose(ms_lr_dict['B'])) + np.power(np.diag(ms_lr_dict['D']),2) , 1e14),
        patterns=ms_lr_pattern, free=True, argnums=1)

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        s = np.sqrt(my_rs.chisquare(df, n_samples) / df)
        param_dict = ms_lr_pattern.fold(var_param)
        z1 = my_rs.randn(n_samples, dim)
        z2 = my_rs.randn(n_samples, M)
        B = param_dict['B']
        D=param_dict['D']
        #D1 = np.sqrt(D)
        return param_dict['mu'] + np.dot(z2, np.transpose(B))/s[:,np.newaxis] + z1*np.transpose(D)

    def mean_and_cov(var_param):
        param_dict = ms_lr_pattern.fold(var_param)
        return param_dict['mu'], df / (df - 2.) * np.dot(param_dict['B'],
                                                         np.transpose(param_dict['B'])) + np.power(np.diag(param_dict['D']),2)

    def entropy(var_param):
        # ignore terms that depend only on df
        param_dict = ms_lr_pattern.fold(var_param)
        Sigma = np.dot(param_dict['B'], np.transpose(param_dict['B'])) + np.power(np.diag(param_dict['D']),2)
        return .5*np.log(np.linalg.det(Sigma))

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = ms_lr_pattern.fold(var_param)
        sq_scales = np.linalg.eigvalsh(param_dict['Sigma'])
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(sq_scales)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(sq_scales**2) + np.sum(sq_scales)**2)

    return VariationalFamily(sample, entropy, logdensity, mean_and_cov,
                             pth_moment, ms_lr_pattern.flat_length(True))



def black_box_klvi(var_family, logdensity, n_samples):
    def variational_objective(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples)
        lower_bound = var_family.entropy(var_param) + np.mean(logdensity(samples))
        return -lower_bound

    objective_and_grad = value_and_grad(variational_objective)

    return objective_and_grad

# This function also computes the gradients per sample for KLVI
def black_box_klvi_all_grads(var_family, logdensity, n_samples, gradient_type='rp'):
    def compute_log_weights(var_param, seed):
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    def compute_objective_mean(var_param, seed):
        log_weights = compute_log_weights(var_param, seed)
        return -np.mean(log_weights)

    def compute_objective_evals(var_param, seed):
        log_weights = compute_log_weights(var_param, seed)
        return -log_weights


    log_weights_vjp = vector_jacobian_product(compute_objective_mean)
    sample_grads= jacobian(compute_objective_evals)

    def objective_grad_and_log_norm(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        seed = npr.randint(2 ** 32)
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        obj_value = compute_objective_mean(var_param, seed)
        obj_grad = log_weights_vjp(var_param, seed, obj_value)
        #particle_grads = sample_grads(var_param, seed, log_weights)
        particle_grads = jacobian(compute_objective_evals)(var_param, seed)
        print(particle_grads.shape)
        _, paretok = psislw(log_weights)
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)
        Neff = np.sum(scaled_values)**2 / np.sum(scaled_values**2)
        return obj_value, obj_grad, particle_grads
    return objective_grad_and_log_norm


# This function also computes the gradients per sample for CHIVI
def black_box_chivi_all_grads(alpha, var_family, logdensity, n_samples):
    def compute_log_weights(var_param, seed):
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    def compute_objective_mean(var_param, seed):
        log_weights = compute_log_weights(var_param, seed)
        log_norm = np.max(log_weights)
        particle_weights = compute_objective_evals(var_param, seed)
        return np.log(np.mean(particle_weights))/alpha + log_norm

    def compute_objective_evals(var_param, seed):
        log_weights = compute_log_weights(var_param, seed)
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)**alpha
        return scaled_values


    log_weights_vjp = vector_jacobian_product(compute_objective_mean)
    sample_grads= jacobian(compute_objective_evals)
    def objective_grad_and_log_norm(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        seed = npr.randint(2 ** 32)
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        particle_grads = jacobian(compute_objective_evals)(var_param, seed)
        print(particle_grads.shape)
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)**alpha
        paretok,_ = gpdfit(scaled_values)
        #print(paretok)
        obj_value = np.log(np.mean(scaled_values))/alpha + log_norm
        obj_grad = alpha*log_weights_vjp(var_param, seed, obj_value)
        return obj_value, obj_grad, particle_grads
    return objective_grad_and_log_norm



def black_box_klvi2(var_family, logdensity, n_samples):
    def compute_log_weights(var_param, seed):
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    def compute_objective_mean(var_param, seed):
        log_weights = compute_log_weights(var_param, seed)
        return -np.mean(log_weights)

    log_weights_vjp = vector_jacobian_product(compute_objective_mean)

    def objective_grad_and_log_norm(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        seed = npr.randint(2 ** 32)
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        obj_value = compute_objective_mean(var_param, seed)
        obj_grad = log_weights_vjp(var_param, seed, obj_value)
        _, paretok = psislw(log_weights)
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)
        Neff = np.sum(scaled_values)**2 / np.sum(scaled_values**2)
        return obj_value, obj_grad,  paretok, Neff
    return objective_grad_and_log_norm


def  black_box_chivi_robust(alpha, var_family, logdensity, S_init=400, c=1.5, Mstar = 10, kstar = 8., S_max=5000,
                            S_min=200):
    def compute_log_weights(var_param, n_samples, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights)

    def compute_ESS(log_weights):
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)
        Neff1 = np.sum(scaled_values)**2 / np.sum(scaled_values**2)
        return Neff1

    def objective_grad_and_log_norm(var_param, S_iter=0):
        if not S_iter:
            S = S_init
        else:
            S = S_iter

        if S < S_min:
            S = S_min
        seed = npr.randint(2**32)
        log_weights = compute_log_weights(var_param, S, seed)
        #_, paretok = psislw(log_weights)
        jitter= 1e-25
        paretok = 0.0
        indx = 0
        while paretok < 0.1:
            indx= indx + 1
            if indx == 4:
                break
            log_weights = log_weights + jitter
            _, paretok =psislw(log_weights)

        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)**alpha
        Neff1 = compute_ESS(log_weights)
        i=0
        samples_changed = False
        while (paretok  > kstar) or (Neff1  <  Mstar):
            i=i+1
            #print(i)
            if S > S_max:
                break
            samples_changed= True
            #seed= seed +1
            S = int(c*S)
            remS = int((c-1)*S)
            # append new weights to the old weights
            if remS == S:
                remS = remS -1

            #log_weights_new = compute_log_weights(var_param, remS, seed)
            #print(log_weights.shape)
            #log_weights = np.concatenate((log_weights.flatten(), log_weights_new.flatten()), axis=0)
            #print(log_weights.shape)
            log_weights = compute_log_weights(var_param, S, seed)
            _, paretok = psislw(log_weights)
            j=0
            while paretok < 0.1:
                j = j+1
                if j == 4:
                    break
                log_weights = log_weights + jitter
                _, paretok = psislw(log_weights)

            log_norm = np.max(log_weights)
            wts = np.exp(log_weights - log_norm)
            scaled_values = wts**alpha
            #print(scaled_values)
            Neff1 = np.sum(scaled_values)**2/np.sum(scaled_values ** 2)

        #print(scaled_values)
        obj_value = np.log(np.mean(scaled_values))/alpha + log_norm
        obj_grad = alpha*log_weights_vjp(var_param, S, seed, scaled_values) / scaled_values.size

        if not samples_changed:
            if paretok < kstar and compute_ESS(log_weights[:int(S/c)]) > Mstar:
                S = int(S/c)
                #print('sample size reduced!')
        return (obj_value, obj_grad, S , paretok, Neff1)
    return objective_grad_and_log_norm



def black_box_klvi_tempered(var_family, logdensity, n_samples, schedule=None):
    '''
    klvi with tempered posterior ...
    :param var_family:
    :param logdensity:
    :param n_samples:
    :param schedule:
    :return:
    '''
    def compute_log_weights(var_param, seed):
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    def compute_objective_mean(var_param, seed):
        log_weights = compute_log_weights(var_param, seed)
        return -np.mean(log_weights)

    log_weights_vjp = vector_jacobian_product(compute_objective_mean)
    def objective_grad_and_log_norm(var_param, epsilon, iter_count=0):
        """Provides a stochastic estimate of the variational lower bound."""
        seed = npr.randint(2 ** 32)
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples)*epsilon - var_family.logdensity(samples, var_param)
        obj_value = compute_objective_mean(var_param, seed)
        obj_grad = log_weights_vjp(var_param, seed, obj_value)
        _, paretok = psislw(log_weights)
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)
        Neff = np.sum(scaled_values)**2 / np.sum(scaled_values**2)
        return obj_value, obj_grad,  paretok, Neff, epsilon
    return objective_grad_and_log_norm


def black_box_klvi_tempered2(var_family, logdensity, S_init=400, c=1.5, Mstar = 10, kstar = 8., S_max=5000,
                            S_min=200):
    def compute_log_weights(var_param, n_samples, epsilon, seed):
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples)*epsilon - var_family.logdensity(samples, var_param)
        return log_weights

    def compute_objective_mean(var_param, n_samples, epsilon, seed):
        log_weights = compute_log_weights(var_param, n_samples, epsilon, seed)
        return -np.mean(log_weights)


    def compute_ESS(log_weights):
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)
        Neff1 = np.sum(scaled_values)**2 / np.sum(scaled_values**2)
        return Neff1

    log_weights_vjp = vector_jacobian_product(compute_objective_mean)

    def objective_grad_and_log_norm(var_param, S_iter=0, epsilon=0, iter_count=0):
        """Provides a stochastic estimate of the variational lower bound."""
        if not S_iter:
            S = S_init
        else:
            S = S_iter

        if S < S_min:
            S = S_min


        seed = npr.randint(2**32)
        log_weights = compute_log_weights(var_param, S, epsilon, seed)
        #_, paretok = psislw(log_weights)
        jitter= 1e-25
        paretok = 0.0
        indx = 0
        while paretok < 0.1:
            indx= indx + 1
            if indx == 2:
                break
            log_weights = log_weights + jitter
            _, paretok =psislw(log_weights)

        log_norm = np.max(log_weights)
        Neff1 = compute_ESS(log_weights)

        i=0
        samples_changed = False
        while (paretok  > kstar) or (Neff1  <  Mstar):
            i=i+1
            if S > S_max:
                break
            samples_changed= True
            #seed= seed +1
            S = int(c*S)
            remS = int((c-1)*S)
            # append new weights to the old weights
            if remS == S:
                remS = remS -1

            #log_weights_new = compute_log_weights(var_param, remS, seed)
            #print(log_weights.shape)
            #log_weights = np.concatenate((log_weights.flatten(), log_weights_new.flatten()), axis=0)
            #print(log_weights.shape)
            log_weights = compute_log_weights(var_param, S, epsilon, seed)
            _, paretok = psislw(log_weights)
            j=0
            while paretok < 0.05:
                j = j+1
                if j == 2:
                    break
                log_weights = log_weights + jitter
                _, paretok = psislw(log_weights)

            log_norm = np.max(log_weights)
            wts = np.exp(log_weights - log_norm)
            Neff1 = np.sum(wts)**2/np.sum(wts ** 2)

        #adapt epsilon here ....
        #print(scaled_values)
        obj_value = -np.mean(log_weights)
        obj_grad = log_weights_vjp(var_param, S, epsilon, seed, obj_value)

        if not samples_changed:
            if paretok < kstar and compute_ESS(log_weights[:int(S/c)]) > Mstar:
                S = int(S/c)
                #print('sample size reduced!')

        return (obj_value, obj_grad, S , epsilon, paretok, Neff1)

    return objective_grad_and_log_norm




def black_box_chivi(alpha, var_family, logdensity, n_samples):
    def compute_log_weights(var_param, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights)

    def objective_grad_and_log_norm(var_param):
        seed = npr.randint(2**32)
        log_weights = compute_log_weights(var_param, seed)
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)**alpha
        paretok,_ = gpdfit(scaled_values)
        #print(paretok)
        obj_value = np.log(np.mean(scaled_values))/alpha + log_norm
        obj_grad = alpha*log_weights_vjp(var_param, seed, scaled_values) / scaled_values.size
        return (obj_value, obj_grad)

    return objective_grad_and_log_norm



def black_box_chivi2(alpha, var_family, logdensity, n_samples):
    def compute_log_weights(var_param, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, 1, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights)

    def objective_grad_and_log_norm(var_param):
        seed = npr.randint(2**32)

        obj=np.zeros(n_samples)
        obj_grad=np.zeros((n_samples, var_param.size))
        for i in range(1,n_samples):
            seed =i
            log_weights = compute_log_weights(var_param, seed)
            a1 = log_weights_vjp(var_param, seed, log_weights)
            grad_w =a1*log_weights
            obj[i] = np.exp(log_weights)**alpha
            obj_grad[i] = obj[i]*a1

        obj_value = np.mean(obj)
        obj_grad = np.mean(obj_grad, axis=0)
        obj_grad = obj_grad*alpha
        return (obj_value, obj_grad)

    return objective_grad_and_log_norm


def markov_score_climbing_cis(var_family, logdensity, n_samples, k, nu=np.inf):
    '''minimizes the inclusive KL  with the MSC-CIS algorithm'''
    def compute_log_weights(var_param, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights


    def log_Normal_density(x, qmu, qsigma):
        return -0.5*np.log(2*np.pi*np.prod(qsigma)) -0.5*np.dot((x-qmu), )


    def dlogq_dmu(x, qmu, qlogsigma):
        qsigma = np.exp(qlogsigma)
        grad=  (x.flatten()-qmu.flatten())/qsigma.flatten()**2
        return -grad


    def dlogq_dsigma(x, qmu, qlogsigma):
        qsigma=np.exp(qlogsigma)
        e = x.flatten()-qmu.flatten()
        s_4 = 1.0/qsigma**4
        grad= -(((qmu.flatten() + qsigma.flatten() - x.flatten()) *
                  (-qmu.flatten() + qsigma.flatten() + x.flatten())) / qsigma.flatten() ** 2)
        #return -0.5/qsigma* + 0.5*s_4*e**2
        return -grad

    def dlogq_dmu_vec(x, qmu, qlogsigma):
        qsigma = np.exp(qlogsigma)
        grad=  (x-qmu.reshape(-1, qmu.size))/qsigma.reshape(-1, qsigma.size)
        return -grad

    def dlogq_dsigma_vec(x, qmu, qlogsigma):
        qsigma = np.exp(qlogsigma)
        qsigma = qsigma.reshape(-1, qsigma.size)
        qmu = qmu.reshape(-1, qmu.size)
        grad= -(((qmu + qsigma - x) *
                  (-qmu + qsigma + x)) / qsigma** 2)
        return -grad


    def logdensity_q(var_param):
        return partial(var_family.logdensity, var_param=var_param)

    def logdensity_q_grad(var_param):
        score_gradient = elementwise_grad(logdensity_q)
        return score_gradient


    def objective_grad_and_log_norm(var_param, prev_zk):
        seed = npr.randint(2**32)
        #log_weights = compute_log_weights(var_param, seed)
        samples = var_family.sample(var_param, n_samples, seed)
        samples_all = np.concatenate((samples, prev_zk))
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        weight_zk = logdensity(prev_zk) - var_family.logdensity(prev_zk, var_param)

        #print(logdensity(prev_zk))
        #print(var_family.logdensity(prev_zk, var_param))
        weight_zk = np.array(weight_zk)

        if log_weights.ndim ==2:
            log_weights = log_weights[0]

        log_weights_all = np.concatenate((log_weights, weight_zk))
        weights_all = np.exp(log_weights_all)
        weights_snis = weights_all /np.sum(weights_all)
        idx_sample = np.random.choice(n_samples +1, 1, p=weights_snis.flatten())
        weight_sample = samples_all[idx_sample]

        obj_grad = logdensity_q_grad(var_param)
        obj_grad = np.concatenate([dlogq_dmu(weight_sample, var_param[:k], var_param[k:]),
                    dlogq_dsigma(weight_sample, var_param[:k], var_param[k:])])
        obj_val = np.mean(weights_all*log_weights_all)
        return (obj_val, obj_grad, weight_sample)


    def objective_grad_and_log_norm_t(var_param, prev_zk):
        seed = npr.randint(2**32)
        #log_weights = compute_log_weights(var_param, seed)
        samples = var_family.sample(var_param, n_samples, seed)
        samples_all = np.concatenate((samples, prev_zk))
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        weight_zk = logdensity(prev_zk) - var_family.logdensity(prev_zk, var_param)
        #print(logdensity(prev_zk))
        #print(var_family.logdensity(prev_zk, var_param))
        weight_zk = np.array(weight_zk)

        if log_weights.ndim ==2:
            log_weights = log_weights[0]
        log_weights_all = np.concatenate((log_weights, weight_zk))
        weights_all = np.exp(log_weights_all)
        weights_snis = weights_all /np.sum(weights_all)
        #print(np.sum(weights_all))
        #print(weights_snis)
        idx_sample = np.random.choice(n_samples +1, 1, p=weights_snis.flatten())
        weight_sample = samples_all[idx_sample]
        #obj_grad = logdensity_q_grad(var_param)
        obj_grad = np.concatenate([dlogq_dmu_t(weight_sample, var_param[:k], var_param[k:], nu),
                    dlogq_dsigma_t(weight_sample, var_param[:k], var_param[k:], nu)])


        obj_val = np.mean(weights_all*log_weights_all)
        return (obj_val, obj_grad, weight_sample)


    def objective_grad_and_log_norm_rb(var_param, prev_zk):
        seed = npr.randint(2**32)
        #log_weights = compute_log_weights(var_param, seed)
        samples = var_family.sample(var_param, n_samples, seed)
        samples_all = np.concatenate((samples, prev_zk))
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        weight_zk = logdensity(prev_zk) - var_family.logdensity(prev_zk, var_param)
        print(weight_zk.shape)
        log_weights_all = np.concatenate((log_weights, weight_zk))
        weights_all = np.exp(log_weights_all)
        weights_snis = weights_all /np.sum(weights_all)
        print(weights_snis)
        idx_sample = np.random.choice(n_samples +1, 1, p=weights_snis.flatten())
        weight_sample = samples_all[idx_sample]
        obj_grad = logdensity_q_grad(var_param)
        var_grad = np.concatenate([dlogq_dmu_vec(samples_all, var_param[:k], var_param[k:]),
                    dlogq_dsigma_vec(samples_all, var_param[:k], var_param[k:])], axis=1)

        obj_val = np.mean(weights_all*log_weights_all)
        obj_grad= weights_snis @ var_grad
        return (obj_val, obj_grad, weight_sample)

    #return objective_grad_and_log_norm_rb
    if nu < np.inf:
        return  objective_grad_and_log_norm_t
    return objective_grad_and_log_norm


def distilled_importance_sampling(var_family, logdensity, n_samples, k, n, omega=0.04):
    '''minimizes the inclusive KL  with the DIS algorithm'''
    def compute_log_weights(var_param, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights


    def dlogq_dmu_vec(x, qmu, qlogsigma):
        qsigma = np.exp(qlogsigma)
        grad=  (x-qmu.reshape(-1, qmu.size))/qsigma.reshape(-1, qsigma.size)
        return -grad


    def dlogq_dsigma_vec(x, qmu, qlogsigma):
        qsigma = np.exp(qlogsigma)
        qsigma = qsigma.reshape(-1, qsigma.size)
        qmu = qmu.reshape(-1, qmu.size)
        grad= -(((qmu + qsigma - x) *
                  (-qmu + qsigma + x)) / qsigma** 2)
        return -grad

    def compute_tis(weights, omega):
        idx = weights > omega
        weights[idx] =omega
        return weights

    def compute_tis2(weights, omega):
        weights_snis = weights/np.sum(weights)
        idx = weights_snis > omega
        weights[idx] =omega
        return weights

    def resample_weights(samples, weights):
        n_samples =samples.shape[0]
        weights_snis = weights/np.sum(weights)
        idx_sample = np.random.choice(n_samples, n, p=weights_snis.flatten())
        samples_ris = samples[idx_sample]
        weights_ris = weights[idx_sample]
        return weights_ris, samples_ris

    def compute_Neff(log_weights):
        weights= np.exp(log_weights)
        a = np.sum(weights**2)
        b= np.sum(weights)**2
        Neff= b/a
        return Neff


    def objective_grad_and_log_norm(var_param, epsilon, delta=0.001):
        seed = npr.randint(2**32)
        #log_weights = compute_log_weights(var_param, seed)
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples)*epsilon - var_family.logdensity(samples, var_param)
        #_, paretok
        Neff= compute_Neff(log_weights)
        j=0
        while Neff >= n:
            epsilon = epsilon - delta
            log_weights = logdensity(samples) * epsilon - var_family.logdensity(samples, var_param)
            Neff= compute_Neff(log_weights)
            j = j+1

        weights_is = np.exp(log_weights)
        weights_snis = weights_is /np.sum(weights_is)
        weights_tis = compute_tis(weights_is, omega)

        weights_ris, samples_ris = resample_weights(samples, weights_tis)
        log_weights_ris= np.log(weights_ris)
        obj_grad = np.concatenate([dlogq_dmu_vec(samples_ris, var_param[:k], var_param[k:]),
                    dlogq_dsigma_vec(samples_ris, var_param[:k], var_param[k:])])
        S = np.sum(weights_tis)
        obj_grad = obj_grad*S/(n_samples*n)
        obj_val = np.mean(weights_ris*log_weights_ris)
        return (obj_val, obj_grad, epsilon)
    return  objective_grad_and_log_norm



def black_box_chivi_neff(alpha, var_family, logdensity, n_samples):
    def compute_log_weights(var_param, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights)
    def objective_grad_and_log_norm(var_param):
        seed = npr.randint(2**32)
        log_weights = compute_log_weights(var_param, seed)
        log_norm = np.max(log_weights)
        wts_normalized = np.exp(log_weights - log_norm)
        scaled_values = np.exp(log_weights - log_norm)**alpha
        neff1 = np.sum(wts_normalized)**2 / np.sum(wts_normalized**2)
        paretok2,_ = gpdfit(scaled_values)
        paretok1, _ = gpdfit(wts_normalized)
        print(f'pareto1:_{paretok1}')
        print(f'pareto2:_{paretok2}')
        print(f'Neff:_{neff1}' )
        neff2 = np.sum(scaled_values)**2/ np.sum(scaled_values**2)
        #print(neff1)
        #print(neff2)
        obj_value = np.log(np.sum(scaled_values)/neff1)/alpha + log_norm
        obj_grad = alpha*log_weights_vjp(var_param, seed, scaled_values) / scaled_values.size
        return (obj_value, obj_grad, paretok1, neff1)

    return objective_grad_and_log_norm


def black_box_chivi_neff2(alpha, var_family, logdensity, M):
    def compute_log_weights(var_param, n_samples, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights)
    def objective_grad_and_log_norm(var_param):
        n_samples = M
        seed = npr.randint(2**32)
        neff1= 0
        prev_neff1= 0
        prev_neff2= 0
        weighted_neff1= 0
        neff2 = 0
        i=0
        while neff1 <= weighted_neff1:
            i=i+1
            n_samples= n_samples + 200
            log_weights = compute_log_weights(var_param, n_samples, seed)
            log_norm = np.max(log_weights)
            wts_normalized = np.exp(log_weights - log_norm)
            scaled_values = wts_normalized**alpha
            prev_neff1 = neff1
            prev_neff2 = neff2
            neff1 = np.sum(wts_normalized) ** 2 / np.sum(wts_normalized ** 2)
            neff2 = np.sum(scaled_values) ** 2 / np.sum(scaled_values ** 2)
            paretok1, _ = gpdfit(wts_normalized)

            if i==1:
                weighted_neff1 = 1.1*neff1
            else:
                weighted_neff1= (0.3*neff1 + 0.7*prev_neff1)*0.96

            if n_samples > 5000:
                break

        obj_value = np.log(np.mean(scaled_values)) / alpha + log_norm
        #obj_value = np.log(np.sum(scaled_values)/neff)/alpha + log_norm
        obj_grad = alpha*log_weights_vjp(var_param, n_samples, seed, scaled_values) / scaled_values.size
        return (obj_value, obj_grad)

    return objective_grad_and_log_norm



def black_box_chivi_iw_reweighting(alpha, var_family, logdensity, M):
    def compute_log_weights(var_param, n_samples, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights)
    def objective_grad_and_log_norm(var_param):
        n_samples = M
        seed = npr.randint(2**32)
        neff1= 0
        neff2 = 0
        while neff1 < int(M/250):
            n_samples= n_samples + 500
            log_weights = compute_log_weights(var_param, n_samples, seed)
            log_norm = np.max(log_weights)
            wts_normalized = np.exp(log_weights - log_norm)

            scaled_values = np.exp(log_weights - log_norm)**alpha
            neff1 = np.sum(wts_normalized) ** 2 / np.sum(wts_normalized ** 2)
            neff2 = np.sum(scaled_values) ** 2 / np.sum(scaled_values ** 2)
            if n_samples > 5000:
                break

        obj_value = np.log(np.mean(scaled_values)) / alpha + log_norm
        #obj_value = np.log(np.sum(scaled_values)/neff)/alpha + log_norm
        obj_grad = alpha*log_weights_vjp(var_param, n_samples, seed, scaled_values) / scaled_values.size
        return (obj_value, obj_grad)

    return objective_grad_and_log_norm



def black_box_klvi_pd(var_family, logdensity, n_samples):
    def variational_objective(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples)
        lower_bound = np.mean(logdensity(samples)) - np.mean(var_family.logdensity(samples, var_param))
        return -lower_bound
    objective_and_grad = value_and_grad(variational_objective)
    #objective_path_val = np.mean(logdensity(Samples))
    return objective_and_grad


def black_box_klvi_pd2(var_family, logdensity, n_samples):
    #a formulation which avoids path derivatives ...
    def variational_objective(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples)
        a = partial(var_family.logdensity,var_param=var_param)
        def nested_fn(samples):
            lower_bound = np.mean(logdensity(samples)) - np.mean(a(samples))
            return -lower_bound
        b= nested_fn(samples)
        return b
    objective_and_grad = value_and_grad(variational_objective)
    #objective_path_val = np.mean(logdensity(Samples))
    return objective_and_grad



def perturbed_black_box_vi(var_family, logdensity, n_samples, fix_vo=False):
    fix_vo =fix_vo
    def variational_objective(var_param):
        #print(var_param.shape)
        lamda, vo = var_param[:-1], var_param[-1]
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(lamda, n_samples, 10)
        log_weights = logdensity(samples) - var_family.logdensity(samples, lamda)
        #log_norm = np.max(log_weights)
        #sum = log_weights - log_norm + var_param[-1]
        sum = log_weights + vo
        obj1 = sum
        obj2 = (sum **2)/2
        obj3 = (sum **3)/6
        obj = obj1 +obj2 + obj3
        #obj = (obj1)
        obj_val = np.mean(obj * np.exp(var_param[-1]))
        return -obj_val

    obj_grad_vjp = vector_jacobian_product(variational_objective)

    def objective_grad_and_log_norm(var_param):
        obj_value = variational_objective(var_param)
        obj_grad = obj_grad_vjp(var_param, obj_value)
        #lamda = var_param[:-1]
        #lamda = lamda - 0.00000001*obj_grad[:-1]
        #new_var_params = np.array(lamda.tolist() + [var_param[-1]])
        #obj_value_next = variational_objective(new_var_params)
        #obj_grad[-1] = obj_grad[-1] + obj_value_ne xt
        if fix_vo:
            obj_grad[-1] = 0.

        return obj_value, obj_grad

    #objective_and_grad = value_and_grad(variational_objective)
    return objective_grad_and_log_norm


def _vectorize_if_needed(f, a, axis=-1):
    if a.ndim > 1:
        return np.apply_along_axis(f, axis, a)
    else:
        return f(a)


def _ensure_2d(a):
    while a.ndim < 2:
        a = a[:,np.newaxis]
    return a


def make_stan_log_density(fitobj):
    @primitive
    def log_density(x):
        return _vectorize_if_needed(fitobj.log_prob, x)
    def log_density_vjp(ans, x):
        return lambda g: _ensure_2d(g) * _vectorize_if_needed(fitobj.grad_log_prob, x)
    defvjp(log_density, log_density_vjp)
    return log_density


def make_stan_log_density_grad(fitobj):
    @primitive
    def grad_log_density(x):
        return _vectorize_if_needed(fitobj.grad_log_prob,x)
    def grad_log_density_vjp(ans, x):
        return lambda g: _ensure_2d(g) * _vectorize_if_needed(fitobj.grad_log_prob, x)
    defvjp(grad_log_density, grad_log_density_vjp)
    return grad_log_density



def learning_rate_schedule(n_iters, learning_rate, learning_rate_end):
    if learning_rate <= 0:
        raise ValueError('learning rate must be positive')
    if learning_rate_end is not None:
        if learning_rate <= learning_rate_end:
            raise ValueError('initial learning rate must be greater than final learning rate')
        # constant learning rate for first quarter, then decay like a/(b + i)
        # for middle half, then constant for last quarter
        b = n_iters*learning_rate_end/(2*(learning_rate - learning_rate_end))
        a = learning_rate*b
        start_decrease_at = n_iters//4
        end_decrease_at = 3*n_iters//4
    for i in range(n_iters):
        if learning_rate_end is None or i < start_decrease_at:
            yield learning_rate
        elif i < end_decrease_at:
            yield a / (b + i - start_decrease_at + 1)
        else:
            yield learning_rate_end



def adagrad_optimize(n_iters, objective_and_grad, init_param,
                     has_log_norm=False, window=10,learning_rate=.01,
                     epsilon=.1, learning_rate_end=None, k=15):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    pareto_k_list = []
    neff_list = []
    prev_z = np.zeros((1,k))

    with tqdm.trange(n_iters) as progress:
        try:
            schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
            for i, curr_learning_rate in zip(progress, schedule):
                prev_variational_param = variational_param
                if has_log_norm == 1:
                    obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                elif has_log_norm == 2:
                    obj_val, obj_grad, paretok, neff = objective_and_grad(variational_param)
                    log_norm = 0.
                    if paretok > 0.25:
                        pareto_k_list.append(paretok)
                        neff_list.append(neff)

                elif has_log_norm == 3:
                    obj_val, obj_grad, prev_z = objective_and_grad(variational_param, prev_z)
                    log_norm = 0.
                else:
                    obj_val, obj_grad = objective_and_grad(variational_param)
                    log_norm= 0.

                value_history.append(obj_val)
                local_grad_history.append(obj_grad)
                local_log_norm_history.append(log_norm)
                log_norm_history.append(log_norm)
                if len(local_grad_history) > window:
                    local_grad_history.pop(0)
                    local_log_norm_history.pop(0)
                grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                scaled_grads = grad_scale[:,np.newaxis]*np.array(local_grad_history)
                accum_sum = np.sum(scaled_grads**2, axis=0)
                variational_param = variational_param - curr_learning_rate*obj_grad/np.sqrt(
                        epsilon + accum_sum)
                if i >= 3*n_iters // 4:
                    variational_param_history.append(variational_param.copy())
                if i % 10 == 0:
                    avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                    progress.set_description(
                        'Average Loss = {:,.5g}'.format(avg_loss))
        except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
            # do not print log on the same line
            progress.close()
        finally:
            progress.close()

    variational_param_history = np.array(variational_param_history)
    variational_param_history_list=[]
    variational_param_history_list.append(variational_param_history)
    variational_param_history_chains = np.stack(variational_param_history_list, axis=0)

    pmz_size = variational_param_history_chains.shape[2]
    optimisation_log = dict()

    khat_iterates = []
    khat_iterates2 = []
    start_stats = n_iters - 3000

    for i in range(pmz_size):
        khat_i = compute_khat_iterates(variational_param_history_chains, 0, i, increasing=True)
        khat_iterates.append(khat_i)

    for j in range(pmz_size):
        khat_i = compute_khat_iterates(variational_param_history_chains, 0, j, increasing=False)
        khat_iterates2.append(khat_i)


    khat_objective,_ = gpdfit(np.array(value_history))
    value_history_neg = [-a for a in value_history]

    khat_objective2, _ = gpdfit(np.array(value_history_neg))
    khat_iterates.append(khat_objective)
    khat_iterates2.append(khat_objective2)
    #khat_iterates.append(khat_objective)

    khat_iterates_array = np.stack(khat_iterates, axis=0)
    khat_iterates_array2 = np.stack(khat_iterates2, axis=0)
    optimisation_log['khat_iterates2'] = khat_iterates_array2
    optimisation_log['khat_iterates'] = khat_iterates_array
    optimisation_log['khat_objective'] = khat_objective
    if has_log_norm == 2:
        optimisation_log['paretok'] = np.array(pareto_k_list)
        optimisation_log['neff'] = np.array(neff_list)

    smoothed_opt_param = np.mean(variational_param_history, axis=0)
    return (smoothed_opt_param, variational_param_history,
            np.array(value_history), np.array(log_norm_history), optimisation_log)



def rmsprop_IA_optimize_with_rhat(n_iters, objective_and_grad, init_param,K,
                        has_log_norm=False, window=500, learning_rate=.01,
                        epsilon=.000001, rhat_window=500, averaging=True, n_optimisers=1,
                        r_mean_threshold=1.15, r_sigma_threshold=1.20, tail_avg_iters=2000,
                        avg_grad_norm=False,
                        learning_rate_end=None):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    averaged_variational_param_history = []
    start_avg_iter = n_iters // 1.3
    sum_grad_norm = 0.
    alpha = 0.99
    scaled_sum_grad_norm = 0.
    variational_param_history_list = []
    averaged_variational_param_history_list = []
    variational_param_list = []
    averaged_variational_param_list = []
    averaged_variational_mean_list = []
    averaged_variational_sigmas_list = []
    pareto_k_list = []
    neff_list = []
    eps= 0.

    #window_size=500
    S=0

    for o in range(n_optimisers):
        variational_param_history = []
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.5
        #variational_param = init_param
        #print(variational_param)
        with tqdm.trange(n_iters) as progress:
            try:
                schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
                for i, curr_learning_rate in zip(progress, schedule):
                    prev_variational_param = variational_param
                    if has_log_norm == 1:
                        obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                    elif has_log_norm ==2:
                        obj_val, obj_grad, paretok ,Seff = objective_and_grad(variational_param)
                        log_norm=0.
                        if paretok > 0.07:
                            pareto_k_list.append(paretok)
                            neff_list.append(Seff)
                    elif has_log_norm == 3:
                        obj_val, obj_grad, S, paretok, Seff = objective_and_grad(variational_param, S)
                        log_norm=0.
                        if paretok > 0.06:
                            pareto_k_list.append(paretok)
                            neff_list.append(Seff)
                    elif has_log_norm == 4:
                        obj_val, obj_grad, paretok, Seff, eps = objective_and_grad(variational_param, eps)
                        #epsilon = tempering_scheme(epsilon, i, curr_learning_rate)
                        eps = eps + 0.001
                        log_norm = 0
                        if paretok > 0.06:
                            pareto_k_list.append(paretok)
                            neff_list.append(Seff)
                    elif has_log_norm == 5:
                        obj_val, obj_grad, S, eps, paretok, Seff = objective_and_grad(variational_param, S, eps)
                        #epsilon = tempering_scheme(epsilon, i, curr_learning_rate)
                        eps = eps + 0.001
                        log_norm = 0
                        if paretok > 0.01:
                            pareto_k_list.append(paretok)
                            neff_list.append(Seff)

                    elif has_log_norm == 6:
                        Seff_prev= Seff
                        obj_val, obj_grad, S, eps, paretok, Seff = objective_and_grad(variational_param, S, eps)
                        #epsilon = tempering_scheme(epsilon, i, curr_learning_rate)
                        eps = adaptive_tempering(eps, Seff, Seff_prev)
                        eps = eps + 0.001
                        log_norm = 0
                        if paretok > 0.05:
                            pareto_k_list.append(paretok)
                            neff_list.append(Seff)

                    else:
                        obj_val, obj_grad = objective_and_grad(variational_param)
                        log_norm = 0
                    value_history.append(obj_val)
                    local_grad_history.append(obj_grad)
                    local_log_norm_history.append(log_norm)
                    log_norm_history.append(log_norm)
                    if len(local_grad_history) > window:
                        local_grad_history.pop(0)
                        local_log_norm_history.pop(0)

                    if has_log_norm:
                        grad_norm = np.exp(log_norm)
                    else:
                        grad_norm = np.sum(obj_grad ** 2, axis=0)
                    if i == 0:
                        if avg_grad_norm:
                            sum_grad_squared = grad_norm
                        else:
                            sum_grad_squared=obj_grad**2
                    else:
                        if avg_grad_norm:
                            sum_grad_squared = grad_norm * alpha + (1. - alpha) * grad_norm
                        else:
                            sum_grad_squared = sum_grad_squared*alpha + (1.-alpha)*obj_grad**2
                    grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                    scaled_grads = grad_scale[:, np.newaxis] * np.array(local_grad_history)
                    accum_sum = np.sum(scaled_grads ** 2, axis=0)
                    scaled_sum_grad_norm = scaled_sum_grad_norm * alpha + (1 - alpha) * accum_sum
                    old_variational_param = variational_param.copy()
                    variational_param = variational_param - curr_learning_rate * obj_grad / np.sqrt(
                        epsilon + sum_grad_squared)
                    # variational_param = variational_param - curr_learning_rate * obj_grad / np.sqrt(epsilon + scaled_sum_grad_norm)
                    variational_param_history.append(old_variational_param)
                    if len(variational_param_history) > 100 * window:
                        variational_param_history.pop(0)
                    if i % 100 == 0:
                        avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                        #print(avg_loss)
                        progress.set_description(
                            'Average Loss = {:,.6g}'.format(avg_loss))


            except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                #pass
                progress.close()

        variational_param_history_array = np.array(variational_param_history)
        variational_param_history_list.append(variational_param_history_array)
        variational_param_list.append(variational_param)

    variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
    rhats = compute_R_hat_adaptive_numpy(variational_param_history_chains, window_size=rhat_window)
    rhat_mean_windows, rhat_sigma_windows = rhats[:,:K], rhats[:,K:]
    rhats_halfway = compute_R_hat_halfway(variational_param_history_chains, interval=100, start=200)

    rhat_mean_windows, rhat_sigma_windows = rhats[:,:K], rhats[:,K:]
    rhat_mean_halfway, rhat_sigma_halfway = rhats_halfway[:, :K], rhats_halfway[:, K:]

    start_swa_m_iters = n_iters - tail_avg_iters
    start_swa_s_iters = start_swa_m_iters


    for ee, w in enumerate(rhat_mean_windows):
        if ee == (rhat_mean_windows.shape[0] - 1):
            continue
    # print(R_hat_window_np[ee])
        if (rhat_mean_windows[ee] < r_mean_threshold).all() and (rhat_mean_windows[ee + 1] < r_mean_threshold).all():
            start_swa_m_iters = ee * rhat_window
            break


    for ee, w in enumerate(rhat_sigma_windows):
        if ee == (rhat_sigma_windows.shape[0] - 1):
            continue
    # print(R_hat_window_np[ee])
        if (rhat_sigma_windows[ee] < r_sigma_threshold).all() and (rhat_sigma_windows[ee + 1] < r_sigma_threshold).all():
            start_swa_s_iters = ee * rhat_window
            break


    start_swa_m_iters = n_iters - tail_avg_iters
    start_swa_s_iters = n_iters - tail_avg_iters
    optimisation_log = dict()
    start_stats = max(start_swa_m_iters, start_swa_s_iters)
    print('start stats:')
    # compute autocorrelation for the chains
    #neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_history_chains, start_stats)
    # compute monte carlo se for chains ..
    mcse_per_chain, mcse_combined_list =  monte_carlo_se(variational_param_history_chains, start_stats)
    pmz_size = variational_param_history_chains.shape[2]
    Neff = np.zeros(pmz_size)
    Rhot = []

    mcmc_se2= []
    khat_iterates =[]
    khat_iterates_min =[]
    khat_iterates2 = []

    for i in range(pmz_size):
        neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_history_chains, start_stats, i)
        #mcse_se_combined = monte_carlo_se2(variational_param_history_chains, start_stats,i)
        Neff[i] = neff
        #mcmc_se2.append(mcse_se_combined)
        Rhot.append(rho_t)
        khat_i = compute_khat_iterates(variational_param_history_chains, start_stats, i, increasing=True)
        khat_iterates.append(khat_i)



    for i in range(pmz_size):
        khat_i2 = compute_khat_iterates(variational_param_history_chains, start_stats, i, increasing=False)
        khat_iterates2.append(khat_i2)


    khat_objective,_ = gpdfit(np.array(value_history))
    khat_iterates.append(khat_objective)
    value_history_neg = [-a for a in value_history]

    khat_objective2, _ = gpdfit(np.array(value_history_neg))
    khat_iterates2.append(khat_objective2)

    rhot_array = np.stack(Rhot, axis=0)
    khat_iterates_array = np.stack(khat_iterates, axis=0)
    khat_iterates_array2 = np.stack(khat_iterates2, axis=0)


    if has_log_norm == 2 or has_log_norm == 3 or has_log_norm == 4:
        optimisation_log['paretok'] = pareto_k_list
        optimisation_log['Seff'] = neff_list


    if has_log_norm == 5:
        optimisation_log['paretok'] = pareto_k_list
        optimisation_log['Seff'] = neff_list

    #mcmc_se2_array = np.stack(mcmc_se2, axis=1)
    for o in range(n_optimisers):
        q_locs_dim = variational_param_history_chains[o,:,:K]
        q_log_sigmas_dim = variational_param_history_chains[o, :, K:]
        q_swa_means_iters, q_swa_mean = stochastic_iterate_averaging(q_locs_dim,
                                                                    start_swa_m_iters)
        q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
                                                                              start_swa_s_iters)
        #averaged_variational_params = np.hstack((q_swa_means_iters, q_swa_log_sigmas_iters))
        #q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
        #                                                                      start_swa_s_iters)
        #if start_swa_s_iters > start_swa_m_iters:
        #    averaged_variational_params = np.hstack((q_swa_means_iters[start_swa_s_iters-start_swa_m_iters:], q_swa_log_sigmas_iters))
        #else:
        #    averaged_variational_params = np.hstack(
        #        (q_swa_means_iters, q_swa_log_sigmas_iters[start_swa_m_iters-start_swa_s_iters:]))
        #averaged_variational_param_list.append(averaged_variational_params)
        averaged_variational_mean_list.append(q_swa_means_iters)
        averaged_variational_sigmas_list.append(q_swa_log_sigmas_iters)

    optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
    optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters
    optimisation_log['r_hat_mean'] = rhat_mean_windows
    optimisation_log['r_hat_sigma'] = rhat_sigma_windows

    optimisation_log['r_hat_mean_halfway'] = rhat_mean_halfway
    optimisation_log['r_hat_sigma_halfway'] = rhat_sigma_halfway
    optimisation_log['neff'] = Neff
    optimisation_log['autocov'] = autocov
    optimisation_log['rhot'] = rhot_array
    optimisation_log['start_stats'] = start_stats
    optimisation_log['mcmc_se'] = mcse_combined_list
    #optimisation_log['mcmc_se2'] = mcmc_se2_array

    optimisation_log['khat_iterates'] = khat_iterates_array
    optimisation_log['khat_iterates2'] = khat_iterates_array2


    return (variational_param, variational_param_history_chains, averaged_variational_mean_list,
            averaged_variational_sigmas_list,
            np.array(value_history), np.array(log_norm_history), optimisation_log)




def adam_IA_optimize_with_rhat(n_iters, objective_and_grad, init_param, K,
                        has_log_norm=False, window=500,  learning_rate=.01,
                        epsilon=.000001, rhat_window=500, averaging=True, n_optimisers=1,
                               r_mean_threshold=1.15, r_sigma_threshold=1.20, learning_rate_end=None):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    averaged_variational_param_history = []
    start_avg_iter = n_iters // 1.3
    sum_grad_norm = 0.
    alpha = 0.9
    scaled_sum_grad_norm = 0.
    variational_param_history_list = []
    averaged_variational_param_history_list = []
    variational_param_list = []
    averaged_variational_param_list = []
    averaged_variational_mean_list = []
    averaged_variational_sigmas_list = []
    window_size=500
    grad_val= 0.
    grad_squared=0
    beta1=0.9
    beta2=0.999
    pareto_k_list = []
    neff_list = []
    r_mean_threshold= 1.10
    r_sigma_threshold = 1.20

    for o in range(n_optimisers):
        variational_param_history = []
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.2
        #variational_param = init_param
        #print(variational_param)
        with tqdm.trange(n_iters) as progress:
            try:
                schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
                for i, curr_learning_rate in zip(progress, schedule):
                    prev_variational_param = variational_param
                    if has_log_norm == 1:
                        obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                    elif has_log_norm ==2:
                        obj_val, obj_grad, paretok ,Seff = objective_and_grad(variational_param)
                        log_norm=0.
                        if paretok > 0.07:
                            pareto_k_list.append(paretok)
                            neff_list.append(Seff)
                    elif has_log_norm == 3:
                        obj_val, obj_grad, S, paretok, Seff = objective_and_grad(variational_param, S)
                        log_norm=0.
                        if paretok > 0.06:
                            pareto_k_list.append(paretok)
                            neff_list.append(Seff)
                    else:
                        obj_val, obj_grad = objective_and_grad(variational_param)
                        log_norm = 0
                    value_history.append(obj_val)
                    local_grad_history.append(obj_grad)
                    local_log_norm_history.append(log_norm)
                    log_norm_history.append(log_norm)
                    if len(local_grad_history) > window:
                        local_grad_history.pop(0)
                        local_log_norm_history.pop(0)

                    if has_log_norm:
                        grad_norm = np.exp(log_norm)
                    else:
                        grad_norm = np.sum(obj_grad ** 2, axis=0)
                    if i == 0:
                        grad_squared = 0.9 * obj_grad ** 2
                        grad_val = 0.9 * obj_grad
                    else:
                        grad_squared = grad_squared * beta2 + (1. - beta2) * obj_grad ** 2
                        grad_val = grad_val * beta1 + (1. - beta1) * obj_grad
                    grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                    scaled_grads = grad_scale[:, np.newaxis] * np.array(local_grad_history)
                    accum_sum = np.sum(scaled_grads ** 2, axis=0)
                    old_variational_param = variational_param.copy()
                    m_hat = grad_val / (1 - np.power(beta1, i + 2))
                    v_hat = grad_squared / (1 - np.power(beta2, i + 2))
                    variational_param = variational_param - curr_learning_rate * m_hat / np.sqrt(epsilon + v_hat)
                    if averaging is True and i > start_avg_iter:
                        averaged_variational_param = (variational_param + old_variational_param * (
                                    i - start_avg_iter)) / (i - start_avg_iter + 1)
                        averaged_variational_param_history.append(averaged_variational_param)
                    variational_param_history.append(old_variational_param)
                    if len(variational_param_history) > 100 * window:
                        variational_param_history.pop(0)
                    if i % 100 == 0:
                        avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                        #print(avg_loss)
                        progress.set_description(
                            'Average Loss = {:,.6g}'.format(avg_loss))


            except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                #pass
                progress.close()

        variational_param_history_array = np.array(variational_param_history)
        variational_param_history_list.append(variational_param_history_array)
        variational_param_list.append(variational_param)

    variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
    rhats = compute_R_hat_adaptive_numpy(variational_param_history_chains, window_size=rhat_window)

    rhats_halfway = compute_R_hat_halfway(variational_param_history_chains, interval=100, start=200)

    rhat_mean_windows, rhat_sigma_windows = rhats[:,:K], rhats[:,K:]
    rhat_mean_halfway, rhat_sigma_halfway = rhats_halfway[:, :K], rhats_halfway[:, K:]

    tail_avg_iters =500
    start_swa_m_iters = n_iters - tail_avg_iters
    start_swa_s_iters = n_iters - tail_avg_iters
    optimisation_log = dict()
    start_stats = max(start_swa_m_iters, start_swa_s_iters)
    print('start stats:')
    # compute autocorrelation for the chains
    #neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_history_chains, start_stats)
    # compute monte carlo se for chains ..
    mcse_per_chain, mcse_combined_list =  monte_carlo_se(variational_param_history_chains, start_stats)
    pmz_size = variational_param_history_chains.shape[2]
    Neff = np.zeros(pmz_size)
    Rhot = []

    mcmc_se2= []
    khat_iterates =[]
    khat_iterates_min =[]
    khat_iterates2 = []

    for i in range(pmz_size):
        neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_history_chains, start_stats, i)
        #mcse_se_combined = monte_carlo_se2(variational_param_history_chains, start_stats,i)
        Neff[i] = neff
        #mcmc_se2.append(mcse_se_combined)
        Rhot.append(rho_t)
        khat_i = compute_khat_iterates(variational_param_history_chains, start_stats, i, increasing=True)
        khat_iterates.append(khat_i)


    for i in range(pmz_size):
        khat_i2 = compute_khat_iterates(variational_param_history_chains, start_stats, i, increasing=False)
        khat_iterates2.append(khat_i2)


    khat_objective,_ = gpdfit(np.array(value_history))
    khat_iterates.append(khat_objective)
    value_history_neg = [-a for a in value_history]

    khat_objective2, _ = gpdfit(np.array(value_history_neg))
    khat_iterates2.append(khat_objective2)

    rhot_array = np.stack(Rhot, axis=0)
    khat_iterates_array = np.stack(khat_iterates, axis=0)
    khat_iterates_array2 = np.stack(khat_iterates2, axis=0)
    khat_combined = np.maximum(khat_iterates, khat_iterates2)


    if has_log_norm == 2 or has_log_norm == 3:
        optimisation_log['paretok'] = pareto_k_list
        optimisation_log['Seff'] = neff_list


    start_swa_m_iters = n_iters - 2000
    start_swa_s_iters = n_iters - 2000
    for ee, w in enumerate(rhat_mean_windows):
        if ee == (rhat_mean_windows.shape[0] - 1):
            continue
    # print(R_hat_window_np[ee])
        if (rhat_mean_windows[ee] < r_mean_threshold).all() and (rhat_mean_windows[ee + 1] < r_mean_threshold).all():
            start_swa_m_iters = ee * rhat_window
            break

    for ee, w in enumerate(rhat_sigma_windows):
        if ee == (rhat_sigma_windows.shape[0] - 1):
            continue
    # print(R_hat_window_np[ee])
        if (rhat_sigma_windows[ee] < r_sigma_threshold).all() and (rhat_sigma_windows[ee + 1] < r_sigma_threshold).all():
            start_swa_s_iters = ee * rhat_window
            break

    optimisation_log = dict()

    for o in range(n_optimisers):
        q_locs_dim = variational_param_history_chains[o,:,:K]
        q_log_sigmas_dim = variational_param_history_chains[o, :, K:]
        q_swa_means_iters, q_swa_mean = stochastic_iterate_averaging(q_locs_dim,
                                                                    start_swa_m_iters)
        q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
                                                                              start_swa_s_iters)

        #averaged_variational_params = np.hstack((q_swa_means_iters, q_swa_log_sigmas_iters))



        #q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
        #                                                                      start_swa_s_iters)

        #if start_swa_s_iters > start_swa_m_iters:
        #    averaged_variational_params = np.hstack((q_swa_means_iters[start_swa_s_iters-start_swa_m_iters:], q_swa_log_sigmas_iters))
        #else:
        #    averaged_variational_params = np.hstack(
        #        (q_swa_means_iters, q_swa_log_sigmas_iters[start_swa_m_iters-start_swa_s_iters:]))
        #averaged_variational_param_list.append(averaged_variational_params)
        averaged_variational_mean_list.append(q_swa_means_iters)
        averaged_variational_sigmas_list.append(q_swa_log_sigmas_iters)

    optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
    optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters

    optimisation_log['r_hat_mean'] = rhat_mean_windows
    optimisation_log['r_hat_sigma'] = rhat_sigma_windows

    optimisation_log['r_hat_mean_halfway'] = rhat_mean_halfway
    optimisation_log['r_hat_sigma_halfway'] = rhat_sigma_halfway
    optimisation_log['neff'] = Neff
    optimisation_log['autocov'] = autocov
    optimisation_log['rhot'] = rhot_array
    optimisation_log['start_stats'] = start_stats
    optimisation_log['mcmc_se'] = mcse_combined_list
    #optimisation_log['mcmc_se2'] = mcmc_se2_array
    optimisation_log['khat_iterates'] = khat_iterates_array
    optimisation_log['khat_iterates2'] = khat_iterates_array2
    optimisation_log['khat_iterates_comb'] = khat_combined
    return (variational_param, variational_param_history_chains, averaged_variational_mean_list,
            averaged_variational_sigmas_list,
            np.array(value_history), np.array(log_norm_history), optimisation_log)


def black_box_fdiv(beta, var_family, logdensity, n_samples):
    def compute_g(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples)
        return samples

    def compute_log_weights(var_param):
        def compute(samples):
            log_weights = logdensity(samples) - var_family.logdensity(samples, getval(var_param))
            return log_weights
        return compute

    def compute_log_weights2(var_param, seed):
        samples = var_family.sample(var_param, n_samples, seed)
        #log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        log_weights = logdensity(samples) - var_family.logdensity(samples, getval(var_param))
        return log_weights

    def compute_objective_evals(var_param, seed):
        log_weights = compute_log_weights2(var_param, seed)
        return -log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights2)

    def f_w(t, w):
        return np.mean(w >= t)


    def objective(var_param, samples):
        logp = logdensity(samples)
        logq = var_family.logdensity(samples, var_param)

        unweighted_obj = np.sum(logq - logp)
        return unweighted_obj


    def phi(logp, logq):
        diff = logp - logq
        norm_diff = diff - np.max(diff)
        dx = np.exp(norm_diff)
        prob = np.sign(dx[np.newaxis, ...] - dx[..., np.newaxis])
        prob = np.array(prob > 0.5, dtype = np.float32)
        wx = np.sum(prob, axis = 1) / len(logp)
        wx = (1. - wx) ** beta
        return wx / np.sum(wx)


    def objective_grad_and_log_norm(var_param, mode=3):

        seed = npr.randint(2 ** 32)
        if mode == 1:
            samples = var_family.sample(var_param, n_samples)
            logp = logdensity(samples)
            logq = var_family.logdensity(samples, var_param)

            ograd = grad(objective, 0)
            obj_grad = ograd(var_param, samples)

            weights = phi(logp, logq)
            obj = np.sum(weights * (logq - logp))
            return obj, obj_grad


        def samples2(var_param):
            samples = var_family.sample(var_param, n_samples, seed)
            return samples

        samples = compute_g(var_param)
        compute_logw = compute_log_weights(var_param)

        #log_weights_vjp = vector_jacobian_product(compute_objective_mean)
        #log_weights, grad_log_weights = value_and_grad(compute_logw, samples)

        particle_grads = elementwise_grad(compute_logw)(samples)
        log_weights = compute_log_weights2(var_param,seed)
        #obj_grad = log_weights_vjp(var_param, seed, obj_value)

        pmz_len = len(var_param)
        n_theta = pmz_len//2

        weights = np.exp(log_weights - np.max(log_weights))
        gamma = np.array([f_w(weights[i], weights) for i in range(len(weights))]) ** beta
        gamma1= gamma[:,None]
        z_gamma = np.sum(gamma1)

        # preferred mode, using autograd gradients, formulating the gradient as in eq
        if mode == 2:
            transform_grad = jacobian(samples2)(var_param)
            a1 = np.transpose(transform_grad, (0,2,1))
            b1 = np.transpose(particle_grads[:,None], (2,1,0))
            #grad_t1 = np.dot(a1,b1)

            l = a1.shape[1]

            grad_t2 = np.zeros((n_samples, l))
            for i in range(l):
                c1 = np.diag(np.dot(a1[:,i,:], b1[:,0,:]))
                grad_t2[:, i] = c1


            obj_grad_all = -(1. / z_gamma) * np.mean(gamma1  * grad_t2, axis=0)
            #print(obj_grad_all.shape)

        if mode == 3:

            obj_grad = -(1. / z_gamma) * gamma1  * particle_grads
            obj_grad_all = np.concatenate([np.mean(obj_grad, axis=0), np.mean(samples*obj_grad*np.exp(var_param[n_theta:]), axis=0)])
            #print(obj_grad_all.shape)
            obj_value = 0
        return (0, obj_grad_all)
    return objective_grad_and_log_norm




# coding: utf-8

# In[1]:
import sys, os


import sys, os
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('../../../..')

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
from  viabel.optimizers_workflow import adagrad_workflow_optimize, rmsprop_workflow_optimize, \
    adam_workflow_optimize
from collections import namedtuple
from data_generation import generate_density

from autograd import value_and_grad, vector_jacobian_product, grad, elementwise_grad
from  viabel.psis import psislw
from  viabel.misc import WeightsSlicer, mog_logprob
from viabel.plot import plot_isocontours


from viabel.vb import adagrad_optimize
import autograd.numpy.random as npr
from  scipy.stats import multivariate_normal
import autograd.scipy.stats.multivariate_normal as mvn
sns.set_style('white')
sns.set_context('notebook', font_scale=2, rc={'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5,5)

from psis import psislw

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--family', type=str, default='gaussian_mf')
parser.add_argument('--divergence', type=int, default=1)
parser.add_argument('--rank', type=str, default='fullrank')
parser.add_argument('--dimensions', type=int, default=50)
parser.add_argument('--covariance', type=str, default='uniform')
parser.add_argument('--iterations', type=int, default = 5000)

import pickle

args = parser.parse_args()
family = args.family
divergence = args.divergence
rank = args.rank
DIM = args.dimensions
cov_structure = args.covariance


import sys, os
sys.path.append('..')
sys.path.append('../..')
#from viabel.vb import low_rank_gaussian_variational_family
from viabel.vb import mean_field_gaussian_variational_family, black_box_klvi, black_box_chivi, adagrad_optimize, t_variational_family


NormalizingFlowConstructor = namedtuple('NormalizingFlowModule',
                                   ['flow', 'flow_det'])


def planar_flow():
    def uhat(u,w):
        what = w/np.dot(w,w)
        #what = w/np.linalg.norm(w,2)
        mwu = -1 + np.log(1 + np.exp(np.dot(w,u)))
        return u + (mwu -np.dot(w,u))*what

    def flow(z,w,b, u):
        h = np.tanh(np.dot(z,w) + b)
        return z + uhat(u, w)*h[:,None]

    def flow_det(z,w,b,u):
        x = np.dot(z,w) +b
        g = elementwise_grad(np.tanh)(np.dot(z,w) + b )[:,None]*w
        return np.abs(1 + np.dot(g, uhat(u,w)))


    return NormalizingFlowConstructor(flow, flow_det)


def black_box_norm_flow(logdensity,D, beta_schedule, num_layers=1, n_samples=100):
    slicer = WeightsSlicer()
    layers = []

    for l in range(num_layers):
        slicer.add_shape("u_%d"%l, (D, ))
        slicer.add_shape("w_%d"%l, (D, ))
        slicer.add_shape("b_%d"%l, (1, ))
        layers.append(("u_%d"%l, "w_%d"%l, "b_%d"%l))

    # make planar transformation functions
    flow, flow_det = planar_flow()
    def forward(z, var_params, layers):
        z_current= z
        ldet_sum = np.zeros(z.shape[0])
        for l, ln in enumerate(layers):
            ul = slicer.get(var_params, ln[0])
            wl = slicer.get(var_params, ln[1])
            bl = slicer.get(var_params, ln[2])
            z_current = flow(z_current, wl, bl, ul)

            ldet_sum = ldet_sum + np.log(flow_det(z_current, wl, bl, ul))

        return  z_current, ldet_sum


    def sample(var_params, n_samples, eps=None, seed=42):
        npr.RandomState(seed)

        if eps is None:
            eps = npr.randn(n_samples, D)

        zs, ldet_sum = forward(eps, var_params, layers)
        return zs, ldet_sum


    def add_extra_layer():
        print('lol')
        num_layers=len(layers)
        slicer.add_shape("u_%d"%num_layers, (D, ))
        slicer.add_shape("w_%d"%num_layers, (D, ))
        slicer.add_shape("b_%d"%num_layers, (1, ))
        layers.append(("u_%d"%num_layers, "w_%d"%num_layers, "b_%d"%num_layers))


    def remove_one_layer():
        layers.pop()


    def get_num_weights():
        return slicer.num_weights

    def qlogprob(var_params, n_samples, eps=None):
        if eps is None:
            eps = npr.randn(n_samples, D)

        zs, ldet_sum = forward(eps, var_params, layers)
        lls = mvn.logpdf(eps, mean=np.zeros(D), cov=np.eye(D)) - ldet_sum
        print(lls.shape)

        return lls, zs

    def compute_k_hat(var_params, n_samples, logdensity, eps=None):
        log_q, zs = qlogprob(var_params, n_samples)
        log_p = logdensity(zs)
        log_weights= log_p - log_q
        _, paretok = psislw(log_weights)
        return paretok


    def get_samples_and_log_weights( var_params, n_samples, logdensity):
        log_q, zs = qlogprob(var_params, n_samples)

        log_p = logdensity(zs)
        log_weights= log_p - log_q
        _, paretok = psislw(log_weights)

        return zs, paretok, log_weights


    def psis_correction():
        samples, log_weights = get_samples_and_log_weights(logdensity, var_family,
                                                           var_param, n_samples)
        smoothed_log_weights, khat = psislw(log_weights)
        return samples.T, smoothed_log_weights, khat



    def lnq_grid(var_params):
        assert D == 2
        xg, yg = np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)
        xx, yy = np.meshgrid(xg, yg)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        zs, ldets = forward(pts, var_params, layers)
        lls = mvn.logpdf(pts, mean=np.zeros(D), cov=np.eye(D)) - ldets
        return zs[:,0].reshape(xx.shape), zs[:,1].reshape(yy.shape), lls.reshape(xx.shape)


    def objective(var_params, t, n_samples, beta=1.):
        # sample from the current approximate density
        zs, ldet_sum = sample(var_params, n_samples)

        ldet_mean = np.mean(ldet_sum)
        loglik = logdensity(zs)
        loglik_mean = np.mean(loglik)

        if t < len(beta_schedule):
            beta = beta_schedule[t]
        else:
            beta=1.

        elbo = beta*loglik_mean + ldet_mean
        return -elbo

    gradient = grad(objective)



    def objective2(var_params, beta=1.):
        t=1
        zs, ldet_sum = sample(var_params, n_samples)
        ldet_mean = np.mean(ldet_sum)
        loglik = logdensity(zs)
        loglik_mean = np.mean(loglik)
        elbo = beta*loglik_mean + ldet_mean
        return -elbo

    objective_and_grad = value_and_grad(objective2)
    return objective, gradient, slicer.num_weights, sample, lnq_grid, compute_k_hat, get_samples_and_log_weights, \
           add_extra_layer, remove_one_layer, get_num_weights, objective_and_grad



def plot_contours(means, covs, colors=None, xlim=[-2.5,2.5], ylim=[-2.5, 2.5], corr=None, savepath=None):
    xlist = np.linspace(xlim[0], xlim[1], 100)
    ylist = np.linspace(ylim[0], ylim[1], 100)
    X,Y = np.meshgrid(xlist, ylist)
    XY = np.concatenate([X[:,:,np.newaxis], Y[:,:,np.newaxis]], axis=2)
    colors = colors or sns.color_palette()
    for m, c, col in zip(means, covs, colors):
        Z = multivariate_normal.pdf(XY, mean=m, cov=c)
        plt.contour(X, Y, Z, colors=[col], linestyles='solid')
    if corr is not None:
        plt.title('correlation = {:.2f}'.format(corr))
    if savepath is None:
        savepath= f'../../writing/variational-objectives/figures_new/{cov_structure}-kl-vb-corr-{corr}.pdf'
        plt.savefig(savepath)
        plt.clf()
    else:
        plt.savefig(savepath, bbox_inches='tight')
        plt.clf()
    #plt.show()


# In[5]:

def gaussianKL(m1, c1, m2, c2):
    delta = m1 - m2
    p2 = np.linalg.inv(c2)
    return .5 * (np.trace(np.dot(c1, p2)) + np.dot(np.dot(delta, p2), delta)
                 - m1.size + np.log(np.linalg.det(c2)) - np.log(np.linalg.det(c1)))



rhos = [0.1, 0.3, .5, 0.75, 0.88, 0.94 ]
ds = np.concatenate([np.arange(2,10, 4), np.arange(10,110,10, dtype=int)]) # np.arange(2,11,2,dtype=int)

n_iters=args.iterations

# log unnormalized distribution to learn

if divergence == 1:
    kl_df = pd.DataFrame(columns=['corr', 'Dimension', 'KL', 'KLMC', 'KLanalytical',
                                  'IncKL', 'IncKLMC', 'IncKLanalytical', 'paretok1', 'paretok2'])
    for rho in rhos:
        for d in ds:
            a = [1.]
            K = 1
            icovs = np.zeros((K, d, d))
            dets, pis = np.zeros(K), np.zeros(K)
            means0 = np.tile([1.] * d, (K, 1))
            if K == 2:
                means0 = np.array([[2.6] * d, [-2.6] * d])

            # means = np.array([[-0.8, 1.0], [1., -1.0]])
            print(means0)
            for k in range(K):
                cov = generate_density(d, cov_structure=cov_structure, rho=rho)
                icovs[k, :, :] = np.linalg.inv(cov)
                signa, logdet = np.linalg.slogdet(cov)
                dets[k] = np.exp(logdet)
                pis[k] = 1. / K

            lnpdf = lambda z: mog_logprob(z, means0, icovs, dets, pis)
            objective, gradient, num_variational_params, sample_z, lnq_grid, compute_k_hat, get_samples_and_log_weights_ncp, add_layer, remove_layer, get_num_weights, objective_and_grad = \
                black_box_norm_flow(logdensity=lnpdf,
                                    beta_schedule=np.linspace(1., 1., 2000),
                                    D=d, num_layers=2)

            init_var_param = .5 * npr.randn(num_variational_params) - 1.

            num_objective_samps_cp = 10
            nf_klvi_var_param_rms, nf_var_param_list_rms, avg_nf_mean_list_rms, avg_nf_sigmas_list_rms, nf_history_rms, _, op_log_mf_rms1 = \
                rmsprop_workflow_optimize(n_iters, objective_and_grad, init_var_param, 50,
                                          learning_rate=.020, learning_rate_end=0.010, n_optimisers=1, stopping_rule=2,
                                          tolerance=0.003, plotting=False)

            k1 = compute_k_hat(nf_klvi_var_param_rms, 2000000, lnpdf)
            kl_df = kl_df.append(dict(corr=rho, Dimension=d, KL=nf_history_rms[-1],
                                      paretok1=k1), ignore_index=True)


    results = dict()
    results['name'] = cov_structure+'-exclusive_kl_nf_planar_vi'
    results['results'] = kl_df
    filename = '../../writing/variational-objectives/pickles/' + results['name'] + '.pkl'
    file1 = open(filename, 'wb')
    pickle.dump(results, file1)



    sns.lineplot(data=kl_df, x='Dimension', y='paretok1', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.axhline(y=0.7, color='blue', linestyle=':')
    plt.axhline(y=1.0, color='red', linestyle=':')
    plt.ylim((-0.1, 1.2))
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-nf_planar_paretok1.pdf', bbox_inches='tight')
    plt.clf()
#plt.show()



if divergence == 2:
    inc_kl_df = pd.DataFrame(columns=['corr', 'Dimension', 'KL', 'KLMC', 'KLanalytical',
                                      'IncKL',  'IncKLMC', 'IncKLanalytical','paretok1', 'paretok2'])

    for rho in rhos:
        for d in ds:
            c2 = generate_density(d, cov_structure=cov_structure, rho=rho)
            m2 = np.zeros(d)
            mf_g_var_family = mean_field_gaussian_variational_family(dim=d)
            def objective(logc1):
                c1 = np.diag(np.exp(logc1))
                return gaussianKL(m2, c2, m2, c1)
            res = minimize(objective, np.ones(d)*0.4, method='BFGS', jac=grad(objective))
            if d == 2:
                plot_contours(means=[m2]*2, covs=[c2, np.diag(np.exp(res.x))],
                              colors=[(0.,0.,0.)]+sns.color_palette(),
                              xlim=[-2.5,2.5], corr=rho, savepath=f'../../writing/variational-objectives/figures_new/{cov_structure}-inckl-vb-corr-{rho}.pdf')

            init_var_param = np.concatenate([m2, np.array(res.x)/2])
            inc_kl_val, paretok1 = compute_KL_estimate_MC(m2, c2, mf_g_var_family, init_var_param, dim=d )
            inc_inc_kl_est, paretok2 = compute_inclusive_KL_MC(m2, c2, mf_g_var_family, init_var_param, dim=d )
            kl_analytical = gaussianKL(m2, np.diag(np.exp(res.x)), m2, c2)
            inc_kl_analytical = gaussianKL(m2, c2, m2, np.diag(np.exp(res.x)) )
            inc_kl_df = inc_kl_df.append(dict(corr=rho, Dimension=d, IncKL=res.fun, KLMC = inc_kl_val, KLanalytical=kl_analytical,
                                      paretok1=paretok1, KL=kl_analytical, IncKLMC= inc_inc_kl_est, IncKLanalytical = inc_kl_analytical,
                                      paretok2=paretok2), ignore_index=True)



    results = dict()
    results['name'] = cov_structure+'-inclusive_kl_gaussian_mf_vi'
    results['results'] = inc_kl_df
    filename = '../../writing/variational-objectives/pickles/' + results['name'] + '.pkl'
    file1 = open(filename, 'wb')
    pickle.dump(results, file1)

    sns.lineplot(data=inc_kl_df, x='Dimension', y='KL', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-kl-inclusive-kl-gaussian-vb-d1.pdf', bbox_inches='tight')
    #plt.show()
    plt.clf()

    sns.lineplot(data=inc_kl_df, x='Dimension', y='IncKL', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Inclusive KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-inclusive-kl-inclusive-kl-gaussian-vb-d1.pdf', bbox_inches='tight')
    #plt.show()
    plt.clf()

    sns.lineplot(data=inc_kl_df, x='Dimension', y='paretok1', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.axhline(y=0.7, color='blue', linestyle=':')
    plt.axhline(y=1.0, color='red', linestyle=':')
    plt.ylim((-0.5, 1.8))
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-inckl-gaussian_mean_field_paretok1.pdf', bbox_inches='tight')
    #plt.show()
    plt.clf()

    sns.lineplot(data=inc_kl_df, x='Dimension', y='paretok2', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-inckl-gaussian_mean_field_paretok2.pdf', bbox_inches='tight')
    plt.clf()


def mog_logprob(x, means, icovs, lndets, pis):
    """ compute the log likelihood according to a mixture of gaussians
        with means  = [mu0, mu1, ... muk]
             icovs  = [C0^-1, ..., CK^-1]
             lndets = ln [|C0|, ..., |CK|]
             pis    = [pi1, ..., piK] (sum to 1)
        at locations given by x = [x1, ..., xN]
    """
    xx = np.atleast_2d(x)
    D  = xx.shape[1]
    centered = xx[:,:,np.newaxis] - means.T[np.newaxis,:,:]
    solved   = np.einsum('ijk,lji->lki', icovs, centered)
    logprobs = - 0.5*np.sum(solved * centered, axis=1) - (D/2.)*np.log(2*np.pi)  - 0.5*lndets + np.log(pis)
    logprob  = scipy.special.logsumexp(logprobs, axis=1)
    if np.isscalar(x) or len(x.shape) == 1:
        return logprob[0]
    else:
        return logprob




if divergence == 3:
    ds = np.concatenate([np.arange(2,10,2), np.arange(10,DIM,10,dtype=int)]) # np.arange(2,11,2,dtype=int)
    chi_df = pd.DataFrame(columns=['corr', 'Dimension', 'chidiv' 'KL', 'KLMC',
                                      'IncKL',  'IncKLMC', 'paretok1', 'paretok2'])

    for rho in rhos:
        for d in ds:
            if cov_structure == 'uniform':
                c2 = rho*np.ones((d,d))
                c2[np.diag_indices_from(c2)] = 1
            elif cov_structure == 'banded':
                c2 = generate_density(d, cov_structure=cov_structure, rho=rho)
            m2 = np.zeros(d)
            lnpdf = lambda x: mvn.logpdf(x, mean=m2, cov=c2);
            mf_g_var_family = mean_field_gaussian_variational_family(dim=d)
            init_var_param = np.concatenate([m2, np.ones(d)*1.5])

            div_objective_and_grad = black_box_chivi(2, mf_g_var_family, lnpdf, n_samples=1000)
            opt_param, var_param_history, val ,_, opt_chivi_log =  adagrad_optimize(3800, div_objective_and_grad, init_var_param, learning_rate=0.006)

            opt_param1= var_param_history[-1]
            m1, c1 = mf_g_var_family.mean_and_cov(opt_param)

            if d == 2:
                plot_contours(means=[m2, m1], covs=[c2, c1],
                              colors=[(0.,0.,0.)]+sns.color_palette(),
                              xlim=[-2.5,2.5], corr=rho, savepath='../../writing/variational-objectives/figures_new/chivi-corr-{:.2f}.pdf'.format(rho))


            inc_kl_val, paretok1 = compute_KL_estimate_MC(m2, c2, mf_g_var_family, opt_param, dim=d )
            inc_inc_kl_est, paretok2 = compute_inclusive_KL_MC(m2, c2, mf_g_var_family, opt_param, dim=d)
            kl_analytical = gaussianKL(m1, c1, m2, c2)
            inc_kl_analytical = gaussianKL(m2, c2, m1, c1 )
            chi_df = chi_df.append(dict(corr=rho, Dimension=d, chidiv=val[-1], KLMC = inc_kl_val, KLanalytical=kl_analytical,
                                      paretok1=paretok1, KL=kl_analytical, IncKLMC= inc_inc_kl_est, IncKLanalytical = inc_kl_analytical,
                                      paretok2=paretok2), ignore_index=True)


    results = dict()
    results['name'] = cov_structure+'-chivi_gaussian_mf_vi'
    results['results'] = chi_df
    filename = '../../writing/variational-objectives/pickles/' + results['name'] + '.pkl'
    file1 = open(filename, 'wb')
    pickle.dump(results, file1)

    sns.lineplot(data=chi_df, x='Dimension', y='KLanalytical', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-chivi-kl-gaussian-vb-d1.pdf', bbox_inches='tight')
    plt.clf()
    #plt.show()


    sns.lineplot(data=chi_df, x='Dimension', y='IncKLanalytical', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Inclusive KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-chivi-inckl-gaussian-vb-d1.pdf', bbox_inches='tight')
    plt.clf()
    #plt.show()


    sns.lineplot(data=chi_df, x='Dimension', y='paretok1', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.axhline(y=0.7, color='blue', linestyle=':')
    plt.axhline(y=1.0, color='red', linestyle=':')
    plt.ylim((-0.5, 1.8))
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-chivi-kl-gaussian_mean_field_paretok1.pdf', bbox_inches='tight')
    plt.clf()
    #plt.show()


    sns.lineplot(data=chi_df, x='Dimension', y='paretok2', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-chivi-inckl-gaussian_mean_field_paretok2.pdf', bbox_inches='tight')
    plt.clf()

    print(chi_df)

from viabel.vb import low_rank_gaussian_variational_family
mf_lr_g_var_family = low_rank_gaussian_variational_family(dim=4, M=2)


if divergence == 4:
    rhos = [ 0.5]
    ds = np.concatenate([np.arange(4,20,100)]) # np.arange(2,11,2,dtype=int)
    M=2
    for rho in rhos:
        for d in ds:
            c2 = rho*np.ones((d,d))
            c2[np.diag_indices_from(c2)] = 1
            m2 = np.zeros(d)
            lnpdf = lambda x: mvn.logpdf(x, mean=m2, cov=c2);
            mf_g_var_family = mean_field_gaussian_variational_family(dim=d)
            b2 = np.ones(d*M)
            d2 = np.ones(d)
            init_var_param = np.concatenate([m2, b2,d2])

            div_objective_and_grad = black_box_chivi(2,mf_lr_g_var_family, lnpdf, n_samples=1000)
            opt_param, var_param_history, val, _ ,_= adagrad_optimize(3400, div_objective_and_grad, init_var_param, learning_rate=0.012)


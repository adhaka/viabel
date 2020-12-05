
# coding: utf-8

# In[1]:

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import autograd.numpy.random as npr
from  scipy.stats import multivariate_normal
import autograd.scipy.stats.multivariate_normal as mvn
sns.set_style('white')
sns.set_context('notebook', font_scale=2, rc={'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5,5)


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--family', type=str, default='gaussian_mf')
parser.add_argument('--divergence', type=int, default=1)
parser.add_argument('--rank', type=str, default='fullrank')
parser.add_argument('--dimensions', type=int, default=50)

import pickle

args = parser.parse_args()
family = args.family
divergence = args.divergence
rank = args.rank
DIM = args.dimensions

from psis import psislw 


import sys, os
sys.path.append('..')
sys.path.append('../..')
from viabel.vb import low_rank_gaussian_variational_family
from viabel.vb import mean_field_gaussian_variational_family, black_box_klvi, black_box_chivi, adagrad_optimize, t_variational_family



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
        savepath= '../writing/variational-objectives/figures_new/kl-vb-corr-{:.2f}.pdf'.format(corr)
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


# In[6]:

def compute_inclusive_KL_MC(m2, c2, var_family, optim_var_params, dim, n_samples=2000000, seed=40):
    L = np.linalg.cholesky(c2)
    my_rs = npr.RandomState(seed)
    samples = np.dot( my_rs.randn(n_samples, dim), L) + m2
    log_weights = mvn.logpdf(samples, m2, c2) - var_family.logdensity(samples, optim_var_params)
    _, paretok = psislw(log_weights)
    div = np.mean(log_weights)
    weights = np.exp(log_weights)
    div1 = -np.mean(weights*log_weights)
    #print('reverse paretok:', paretok)
    return div, paretok

def compute_KL_estimate_MC(m2, c2, var_family, optim_var_params, dim, n_samples=2000000, seed=40):
    samples = var_family.sample(optim_var_params, n_samples, seed)
    log_weights = mvn.logpdf(samples, m2, c2) - var_family.logdensity(samples, optim_var_params)
    _, paretok = psislw(log_weights)
    div = -np.mean(log_weights)
    #print('kl paretok:', paretok)
    return div, paretok  


# In[7]:

rhos = [0.1, .5, 0.75, 0.88, 0.94, 0.97 ]
# rhos.reverse()
ds = np.concatenate([np.arange(2,10,2), np.arange(10,DIM,10,dtype=int)]) # np.arange(2,11,2,dtype=int)



if divergence == 1:
    kl_df = pd.DataFrame(columns=['corr', 'Dimension', 'KL', 'KLMC', 'KLanalytical',
                                  'IncKL', 'IncKLMC', 'IncKLanalytical', 'paretok1', 'paretok2'])
    for rho in rhos:
        for d in ds:
            c2 = rho*np.ones((d,d))
            c2[np.diag_indices_from(c2)] = 1
            m2 = np.zeros(d)
            mf_g_var_family = mean_field_gaussian_variational_family(dim=d)
            def objective(logc1):
                c1 = np.diag(np.exp(logc1))
                return gaussianKL(m2, c1, m2, c2)
            res = minimize(objective, np.zeros(d), method='BFGS', jac=grad(objective))
            if d == 2:
                plot_contours(means=[m2]*2, covs=[c2, np.diag(np.exp(res.x))],
                              colors=[(0.,0.,0.)]+sns.color_palette(),
                              xlim=[-2.5,2.5], corr=rho)

            init_var_param = np.concatenate([m2, np.array(res.x)/2])
            kl_val, paretok1 = compute_KL_estimate_MC(m2, c2, mf_g_var_family, init_var_param, dim=d )
            kl_analytical = gaussianKL(m2, np.diag(np.exp(res.x)), m2, c2)
            inc_kl_analytical = gaussianKL(m2, c2, m2, np.diag(np.exp(res.x)) )

            inc_kl_est, paretok2 = compute_inclusive_KL_MC(m2, c2, mf_g_var_family, init_var_param, dim=d )
            kl_df = kl_df.append(dict(corr=rho, Dimension=d, KL=res.fun, KLMC = kl_val, KLanalytical=kl_analytical,
                                      paretok1=paretok1, IncKL=inc_kl_analytical, IncKLMC= inc_kl_est, IncKLanalytical = inc_kl_analytical,
                                      paretok2=paretok2), ignore_index=True)


    results = dict()
    results['name'] = 'exclusive_kl_gaussian_mf_vi'
    results['results'] = kl_df
    filename = '../writing/pickles/' + results['name'] + '.pkl'
    file1 = open(filename, 'wb')
    pickle.dump(results, file1)

    sns.lineplot(data=kl_df, x='Dimension', y='KL', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/kl-vb-d1.pdf', bbox_inches='tight')
    plt.clf()    
#plt.show()


    sns.lineplot(data=kl_df, x='Dimension', y='IncKL', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Inclusive KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/inckl-vb-d1.pdf', bbox_inches='tight')
    plt.clf() 
    #plt.show()


    sns.lineplot(data=kl_df, x='Dimension', y='paretok1', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.axhline(y=0.7, color='blue', linestyle=':')
    plt.axhline(y=1.0, color='red', linestyle=':')
    plt.ylim((-0.1, 1.2))
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/gaussian_mean_field_paretok1.pdf', bbox_inches='tight')
    plt.clf()
#plt.show()


    sns.lineplot(data=kl_df, x='Dimension', y='paretok2', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/kl-gaussian_mean_field_paretok2.pdf', bbox_inches='tight')
    plt.clf()


if divergence == 2:
    inc_kl_df = pd.DataFrame(columns=['corr', 'Dimension', 'KL', 'KLMC', 'KLanalytical',
                                      'IncKL',  'IncKLMC', 'IncKLanalytical','paretok1', 'paretok2'])

    for rho in rhos:
        for d in ds:
            c2 = rho*np.ones((d,d))
            c2[np.diag_indices_from(c2)] = 1
            m2 = np.zeros(d)
            mf_g_var_family = mean_field_gaussian_variational_family(dim=d)
            def objective(logc1):
                c1 = np.diag(np.exp(logc1))
                return gaussianKL(m2, c2, m2, c1)
            res = minimize(objective, np.ones(d)*0.4, method='BFGS', jac=grad(objective))
            if d == 2:
                plot_contours(means=[m2]*2, covs=[c2, np.diag(np.exp(res.x))],
                              colors=[(0.,0.,0.)]+sns.color_palette(),
                              xlim=[-2.5,2.5], corr=rho, savepath='../writing/variational-objectives/figures_new/inckl-vb-corr-{:.2f}.pdf'.format(rho))

            init_var_param = np.concatenate([m2, np.array(res.x)/2])
            inc_kl_val, paretok1 = compute_KL_estimate_MC(m2, c2, mf_g_var_family, init_var_param, dim=d )
            inc_inc_kl_est, paretok2 = compute_inclusive_KL_MC(m2, c2, mf_g_var_family, init_var_param, dim=d )
            kl_analytical = gaussianKL(m2, np.diag(np.exp(res.x)), m2, c2)
            inc_kl_analytical = gaussianKL(m2, c2, m2, np.diag(np.exp(res.x)) )
            inc_kl_df = inc_kl_df.append(dict(corr=rho, Dimension=d, IncKL=res.fun, KLMC = inc_kl_val, KLanalytical=kl_analytical,
                                      paretok1=paretok1, KL=kl_analytical, IncKLMC= inc_inc_kl_est, IncKLanalytical = inc_kl_analytical,
                                      paretok2=paretok2), ignore_index=True)



    results = dict()
    results['name'] = 'inclusive_kl_gaussian_mf_vi'
    results['results'] = inc_kl_df
    filename = '../writing/pickles/' + results['name'] + '.pkl'
    file1 = open(filename, 'wb')
    pickle.dump(results, file1)

    sns.lineplot(data=inc_kl_df, x='Dimension', y='KL', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/kl-inclusive-kl-gaussian-vb-d1.pdf', bbox_inches='tight')
    #plt.show()
    plt.clf()

    sns.lineplot(data=inc_kl_df, x='Dimension', y='IncKL', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Inclusive KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/inclusive-kl-inclusive-kl-gaussian-vb-d1.pdf', bbox_inches='tight')
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
    plt.savefig('../writing/variational-objectives/figures_new/inckl-gaussian_mean_field_paretok1.pdf', bbox_inches='tight')
    #plt.show()
    plt.clf()

    sns.lineplot(data=inc_kl_df, x='Dimension', y='paretok2', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/inckl-gaussian_mean_field_paretok2.pdf', bbox_inches='tight')
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
    logprobs = - 0.5*np.sum(solved * centered, axis=1) - (D/2.)*np.log(2*np.pi)                - 0.5*lndets + np.log(pis)
    logprob  = scipy.special.logsumexp(logprobs, axis=1)
    if np.isscalar(x) or len(x.shape) == 1:
        return logprob[0]
    else:
        return logprob




if divergence == 3:
    ds = np.concatenate([np.arange(2,10,2), np.arange(10,DIM,20,dtype=int)]) # np.arange(2,11,2,dtype=int)
    chi_df = pd.DataFrame(columns=['corr', 'Dimension', 'chidiv' 'KL', 'KLMC',
                                      'IncKL',  'IncKLMC', 'paretok1', 'paretok2'])

    for rho in rhos:
        for d in ds:
            c2 = rho*np.ones((d,d))
            c2[np.diag_indices_from(c2)] = 1
            m2 = np.zeros(d)
            lnpdf = lambda x: mvn.logpdf(x, mean=m2, cov=c2);
            mf_g_var_family = mean_field_gaussian_variational_family(dim=d)
            init_var_param = np.concatenate([m2, np.ones(d)*1.5])

            div_objective_and_grad = black_box_chivi(2, mf_g_var_family, lnpdf, n_samples=500)
            opt_param, var_param_history, val, _ ,_=  adagrad_optimize(3600, div_objective_and_grad, init_var_param, learning_rate=0.012)
            if d == 2:
                plot_contours(means=[m2]*2, covs=[c2, np.diag(np.exp(res.x))],
                              colors=[(0.,0.,0.)]+sns.color_palette(),
                              xlim=[-2.5,2.5], corr=rho, savepath='../writing/variational-objectives/figures_new/chivi-corr-{:.2f}.pdf'.format(rho))


            init_var_param = np.concatenate([m2, np.array(res.x)/2])
            inc_kl_val, paretok1 = compute_KL_estimate_MC(m2, c2, mf_g_var_family, init_var_param, dim=d )
            inc_inc_kl_est, paretok2 = compute_inclusive_KL_MC(m2, c2, mf_g_var_family, init_var_param, dim=d )
            kl_analytical = gaussianKL(m2, np.diag(np.exp(res.x)), m2, c2)
            inc_kl_analytical = gaussianKL(m2, c2, m2, np.diag(np.exp(res.x)) )
            chi_df = chi_df.append(dict(corr=rho, Dimension=d, chidiv=val[-1], KLMC = inc_kl_val, KLanalytical=kl_analytical,
                                      paretok1=paretok1, KL=kl_analytical, IncKLMC= inc_inc_kl_est, IncKLanalytical = inc_kl_analytical,
                                      paretok2=paretok2), ignore_index=True)


    results = dict()
    results['name'] = 'chivi_gaussian_mf_vi'
    results['results'] = chi_df
    filename = '../writing/pickles/' + results['name'] + '.pkl'
    file1 = open(filename, 'wb')
    pickle.dump(results, file1)

    sns.lineplot(data=chi_df, x='Dimension', y='KL', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/kl-chivi-gaussian-vb-d1.pdf', bbox_inches='tight')
    #plt.show()


    sns.lineplot(data=chi_df, x='Dimension', y='IncKL', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Inclusive KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/inckl-chivi-gaussian-vb-d1.pdf', bbox_inches='tight')
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
    plt.savefig('../writing/variational-objectives/figures_new/inckl-chivi-gaussian_mean_field_paretok1.pdf', bbox_inches='tight')
    #plt.show()


    sns.lineplot(data=chi_df, x='Dimension', y='paretok2', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('paretok')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig('../writing/variational-objectives/figures_new/inckl-chivi-gaussian_mean_field_paretok2.pdf', bbox_inches='tight')

    print(chi_df)

from viabel.vb import low_rank_gaussian_variational_family
mf_lr_g_var_family = low_rank_gaussian_variational_family(dim=4, M=2)


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
        opt_param, var_param_history, val, _ ,_=         adagrad_optimize(3400, div_objective_and_grad, init_var_param, learning_rate=0.012)

    if d == 2:
        plot_contours(means=[m2, mean], covs=[c2, cov],
                                  colors=[(0.,0.,0.)]+sns.color_palette(),
                                  xlim=[-2.5,2.5], corr=rho)

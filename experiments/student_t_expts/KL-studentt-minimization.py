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

from data_generation import generate_density

from paragami import (PatternDict,
                      NumericVectorPattern,
                      PSDSymmetricMatrixPattern,
                      FlattenFunctionInput)

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

from viabel.vb import (mean_field_gaussian_variational_family,
                       mean_field_t_variational_family,
                       t_variational_family,
                       black_box_klvi,
                       black_box_chivi,
                       make_stan_log_density,
                       _get_mu_sigma_pattern,
                       adagrad_optimize,
                       markov_score_climbing_cis
                      )


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


def plot_approx_and_exact_contours(logdensity, var_family, var_param,colors=None, 
                                    xlim=[-2.5,2.5], ylim=[-2.5, 2.5],
                                    savepath=None, aux_var=None, corr=None):
    xlist = np.linspace(*xlim, 100)
    ylist = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(xlist, ylist)
    XY = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    if aux_var is not None:
        a1= XY.shape[0]
        XY = np.concatenate([XY, np.repeat(aux_var[None,:], a1, axis=0)], axis=1)
    zs = np.exp(logdensity(XY))
    Z = zs.reshape(X.shape)
    zsapprox = np.exp(var_family.logdensity(XY, var_param))
    Zapprox = zsapprox.reshape(X.shape)
    colors = colors or sns.color_palette()
    plt.contour(X, Y, Z, colors=[colors[0]], linestyles='solid')
    plt.contour(X, Y, Zapprox, colors=[colors[2]], linestyles='solid')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.title('correlation = {:.2f}'.format(corr))

    plt.savefig(savepath,
                    bbox_inches='tight')

    plt.clf()

    #plt.show()

lims = dict(xlim=[-2.5,2.5], ylim=[-2.5,2.5])


def compute_inclusive_KL(m2, c2, var_family, optim_var_params, dim, n_samples=2000000, seed=40):
    L = np.linalg.cholesky(c2)
    my_rs = npr.RandomState(seed)
    samples = np.dot( my_rs.randn(n_samples, dim), L) + m2
    log_weights = mvn.logpdf(samples, m2, c2) - var_family.logdensity(samples, optim_var_params)
    _, paretok = psislw(log_weights)
    div = np.mean(log_weights)
    weights = np.exp(log_weights)
    div1 = -np.mean(weights*log_weights)
    print('reverse paretok:', paretok)
    return div, paretok

def compute_KL_estimate(m2, c2, var_family, optim_var_params, dim, n_samples=2000000, seed=40):
    samples = var_family.sample(optim_var_params, n_samples, seed)
    log_weights = mvn.logpdf(samples, m2, c2) - var_family.logdensity(samples, optim_var_params)
    _, paretok = psislw(log_weights)
    div = -np.mean(log_weights)
    print('kl paretok:', paretok)
    return div, paretok  


rhos = [0.1, .5, 0.75, 0.88, 0.94, 0.97 ]
ds = np.concatenate([np.arange(2,10,2), np.arange(10,DIM,10,dtype=int)]) # np.arange(2,11,2,dtype=int)
n_iters = 5000


if divergence==1:
    kl_df = pd.DataFrame(columns=['corr', 'Dimension', 'KL', 'KLMC', 'KLanalytical',
                                  'IncKL', 'IncKLMC', 'IncKLanalytical', 'paretok1', 'paretok2'])
    for rho in rhos:
        for d in ds:
            c2 = generate_density(d, cov_structure=cov_structure, rho=rho)
            m2 = np.zeros(d)

            init_log_std = np.ones(d)*1.5
            init_var_param1 = np.concatenate([m2, init_log_std])
            mf_t_var_family = mean_field_t_variational_family(d, df=8)

            #ms_pattern = _get_mu_sigma_pattern(d)
            #lnpdf2 = FlattenFunctionInput(
            #lambda x: multivariate_t_logpdf(x, m2, c2, 100000),
            #patterns=ms_pattern, free=True, argnums=1)
            lnpdf = lambda z: mvn.logpdf(z, m2, c2)
            #lnpdf_t = lambda z:

            klvi_objective_and_grad = black_box_klvi(mf_t_var_family, lnpdf, 1000)
            klvi_var_param,  klvi_param_history, value_history, grad_norm_history, oplog = adagrad_optimize(5000, klvi_objective_and_grad, init_var_param1, learning_rate=.008,
                                      learning_rate_end=0.001)
            if d == 2:
                plot_approx_and_exact_contours(lnpdf, mf_t_var_family, klvi_var_param, colors=[(0.,0.,0.)]+sns.color_palette()+sns.color_palette(),**lims, savepath=f'../../writing/variational-objectives/figures_new/{cov_structure}-klvi_gauss_vs_t_2D_'+str(rho)+'.pdf',corr=rho )

            kl_val, paretok1 = compute_KL_estimate(m2, c2, mf_t_var_family, klvi_var_param, dim=d )
            inc_kl, paretok2 = compute_inclusive_KL(m2, c2, mf_t_var_family, klvi_var_param, dim=d )
            kl_df = kl_df.append(dict(corr=rho, Dimension=d, KL=value_history[-1], KLMC = kl_val, IncKLMC = inc_kl,  paretok1=paretok1, paretok2=paretok2), ignore_index=True)


    results = dict()
    results['name'] = cov_structure+'-exclusive_kl_t_mf_vi'
    results['results'] = kl_df
    filename = '../../writing/variational-objectives/pickles/' + results['name'] + '.pkl'
    file1 = open(filename, 'wb')
    pickle.dump(results, file1)

    sns.lineplot(data=kl_df, x='Dimension', y='KL', hue='corr', legend='full')
    plt.ylabel('KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-kl-studentt-vb-d1.pdf', bbox_inches='tight')
    plt.clf()

    sns.lineplot(data=kl_df, x='Dimension', y='IncKL', hue='corr', legend='full')
    plt.ylabel('Inclusive KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-inckl-studentt-vb-d1.pdf', bbox_inches='tight')
    plt.clf()

    sns.lineplot(data=kl_df, x='Dimension', y='paretok1', hue='corr', legend='full')
    plt.ylabel('paretok')
    plt.axhline(y=0.7, color='blue', linestyle=':')
    plt.axhline(y=1.0, color='red', linestyle=':')
    plt.ylim((-0.5, 1.8))
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-studentt_mean_field_paretok1.pdf', bbox_inches='tight')
    plt.clf()


    sns.lineplot(data=kl_df, x='Dimension', y='paretok2', hue='corr', legend='full')
    plt.ylabel('paretok')
    plt.ylim((-0.5, 1.8))
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-kl-studentt_mean_field_paretok2.pdf', bbox_inches='tight')
    plt.clf()


if divergence == 2:
    inc_kl_df_t = pd.DataFrame(columns=['corr', 'Dimension', 'KL', 'KLMC', 'KLanalytical',
                                      'IncKL',  'IncKLMC', 'IncKLanalytical','paretok1', 'paretok2'])
    for rho in rhos:
        for d in ds:
            c2 = generate_density(d, cov_structure=cov_structure, rho=rho)
            m2 = np.zeros(d)
            init_log_std = np.ones(d)*1.5
            init_var_param1 = np.concatenate([m2, init_log_std])
            mf_t_var_family = mean_field_t_variational_family(d, df=8)
            lnpdf = lambda z: mvn.logpdf(z, m2, c2)
            incl_klvi_mf_objective_and_grad = markov_score_climbing_cis(mf_t_var_family, lnpdf, 2000,d)


            inc_klvi_var_param, inc_klvi_param_history, obj_history,  inc_klvi_history, op_log_inklvi = adagrad_optimize(n_iters, incl_klvi_mf_objective_and_grad, init_var_param1, learning_rate=.01, learning_rate_end=0.001, has_log_norm=3,k=d)
            if d == 2:
                plot_approx_and_exact_contours(lnpdf, mf_t_var_family, inc_klvi_var_param, colors=[(0.,0.,0.)]+sns.color_palette()+sns.color_palette(),**lims, savepath='../../writing/variational-objectives/figures_new/inclusive_klvi_gauss_vs_t_2D_'+str(rho)+'.pdf',corr=rho)
            kl_val, paretok1 = compute_KL_estimate(m2, c2, mf_t_var_family, inc_klvi_var_param, dim=d )
            inc_kl, paretok2 = compute_inclusive_KL(m2, c2, mf_t_var_family, inc_klvi_var_param, dim=d )
            inc_kl_df_t = inc_kl_df_t.append(dict(corr=rho, Dimension=d, KL=inc_kl, KLMC=kl_val, IncKLMC=inc_kl, paretok1=paretok1, paretok2=paretok2), ignore_index=True)


    results = dict()
    results['name'] = cov_structure+'-inclusive_kl_gaussian_mf_vi'
    results['results'] = inc_kl_df_t
    filename = '../../writing/variational-objectives/pickles/' + results['name'] + '.pkl'
    file1 = open(filename, 'wb')
    pickle.dump(results, file1)

    sns.lineplot(data=inc_kl_df_t, x='Dimension', y='KLMC', hue='corr', legend='full')
    plt.ylabel('KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-kl-inclusive-kl-studentt-vb-d1.pdf', bbox_inches='tight')
    plt.clf()

    sns.lineplot(data=inc_kl_df_t, x='Dimension', y='IncKLMC', hue='corr', legend='full')
    plt.ylabel('Inclusive KL divergence')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-inclusive-kl-inclusive-kl-studentt-vb-d1.pdf', bbox_inches='tight')
    plt.clf()

    sns.lineplot(data=inc_kl_df_t, x='Dimension', y='paretok1', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Inclusive KL divergence')
    plt.axhline(y=0.7, color='blue', linestyle=':')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../../writing/variational-objectives/figures_new/{cov_structure}-inc_kl_studentt_paretok1.pdf', bbox_inches='tight')
    plt.clf()


    sns.lineplot(data=inc_kl_df_t, x='Dimension', y='paretok2', hue='corr', legend='full')
    #plt.legend(rhos, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Paretok')
    plt.axhline(y=0.7, color='blue', linestyle=':')
    plt.legend(rhos, loc='upper center', bbox_to_anchor=(0.5, 1.4),
               ncol=3, frameon=False)
    sns.despine()
    plt.savefig(f'../writing/variational-objectives/figures_new/{cov_structure}-inc_kl_studentt_paretok2.pdf', bbox_inches='tight')
    plt.clf()


if divergence == 3:
    ds = np.concatenate([np.arange(2,DIM,2), np.arange(10,DIM,10,dtype=int)]) # np.arange(2,11,2,dtype=int)
    chi_df = pd.DataFrame(columns=['corr', 'Dimension', 'chidiv' 'KL', 'KLMC',
                                      'IncKL',  'IncKLMC', 'paretok1', 'paretok2'])

    for rho in rhos:
        for d in ds:
            c2 = generate_density(d, cov_structure=cov_structure, rho=rho)
            m2 = np.zeros(d)
            lnpdf = lambda x: mvn.logpdf(x, mean=m2, cov=c2);
            mf_g_var_family = mean_field_gaussian_variational_family(dim=d)
            init_var_param = np.concatenate([m2, np.ones(d)*1.5])

            div_objective_and_grad = black_box_chivi(2, mf_g_var_family, lnpdf, n_samples=1000)
            opt_param, var_param_history, val , _=  adagrad_optimize(5000, div_objective_and_grad, init_var_param, learning_rate=0.006)

            opt_param1= var_param_history[-1]
            m1, c1 = mf_g_var_family.mean_and_cov(opt_param)

            if d == 2:
                plot_contours(means=[m2, m1], covs=[c2, c1],
                              colors=[(0.,0.,0.)]+sns.color_palette(),
                              xlim=[-2.5,2.5], corr=rho, savepath='../../writing/variational-objectives/figures_new/chivi-corr-{:.2f}.pdf'.format(rho))


            inc_kl_val, paretok1 = compute_KL_estimate(m2, c2, mf_g_var_family, var_param_history[-1], dim=d )
            inc_inc_kl_est, paretok2 = compute_inclusive_KL(m2, c2, mf_g_var_family, var_param_history[-1], dim=d)

            chi_df = chi_df.append(dict(corr=rho, Dimension=d, chidiv=val[-1], KLMC = inc_kl_val,
                                      paretok1=paretok1, IncKLMC= inc_inc_kl_est,
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






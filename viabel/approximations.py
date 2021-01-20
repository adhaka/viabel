from abc import ABC, abstractmethod

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist
from autograd.scipy.linalg import sqrtm
from scipy.linalg import eigvalsh

from paragami import (PatternDict,
                      NumericVectorPattern,
                      PSDSymmetricMatrixPattern,
                      FlattenFunctionInput,
                      NumericArrayPattern)

from autograd import value_and_grad, vector_jacobian_product, grad, elementwise_grad
from ._distributions import multivariate_t_logpdf
from ._psis import psislw

from collections import namedtuple

__all__ = [
    'ApproximationFamily',
    'MFGaussian',
    'MFStudentT',
    'MultivariateT',
    'LowRankGaussian'
]

class ApproximationFamily(ABC):
    """An abstract class for an variational approximation family.

    See derived classes for examples.
    """
    def __init__(self, dim, var_param_dim, supports_entropy, supports_kl):
        """
        Parameters
        ----------
        dim : `int`
            The dimension of the space the distributions in the approximation family are defined on.
        var_param_dim : `int`
            The dimension of the variational parameter.
        supports_entropy : `bool`
            Whether the approximation family supports closed-form entropy computation.
        supports_kl : `bool`
            Whether the approximation family supports closed-form KL divergence
            computation.
        """
        self._dim = dim
        self._var_param_dim = var_param_dim
        self._supports_entropy = supports_entropy
        self._supports_kl = supports_kl

    def init_param(self):
        """A variational parameter to use for initialization.

        Returns
        -------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
        """
        return np.zeros(self.var_param_dim)

    @abstractmethod
    def sample(self, var_param, n_samples, seed=None):
        """Generate samples from the variational distribution

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        n_samples : `int`
            The number of samples to generate.

        Returns
        -------
        samples : `numpy.ndarray`, shape (n_samples, var_param_dim)
        """
        pass

    def entropy(self, var_param):
        """Compute entropy of variational distribution.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.

        Raises
        ------
        NotImplementedError
            If entropy computation is not supported."""
        if self.supports_entropy:
            return self._entropy(var_param)
        raise NotImplementedError()

    def _entropy(self, var_param):
        raise NotImplementedError()

    @property
    def supports_entropy(self):
        """Whether the approximation family supports closed-form entropy computation."""
        return self._supports_entropy

    def kl(self, var_param0, var_param1):
        """Compute the Kullback-Leibler (KL) divergence.

        Parameters
        ----------
        var_param0, var_param1 : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameters.

        Raises
        ------
        NotImplementedError
            If KL divergence computation is not supported.
        """
        if self.supports_kl:
            return self._kl(var_param0, var_param1)
        raise NotImplementedError()

    def _kl(self, var_param):
        raise NotImplementedError()

    @property
    def supports_kl(self):
        """Whether the approximation family supports closed-form KL divergence computation."""
        return self._supports_kl

    @abstractmethod
    def log_density(self, var_param, x):
        """The log density of the variational distribution.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        x : `numpy.ndarray`, shape (dim,)
            Value at which to evaluate the density."""
        pass

    @abstractmethod
    def mean_and_cov(self, var_param):
        """The mean and covariance of the variational distribution.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        """
        pass

    def pth_moment(self, var_param, p):
        """The absolute pth moment of the variational distribution.

        The absolute pth moment is given by :math:`\\mathbb{E}[|X|^p]`.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        p : `int`

        Raises
        ------
        ValueError
            If `p` value not supported"""
        if self.supports_pth_moment(p):
            return self._pth_moment(var_param, p)
        raise ValueError('p = {} is not a supported moment'.format(p))

    @abstractmethod
    def _pth_moment(self, var_param, p):
        """Get pth moment of the approximating distribution"""
        pass

    @abstractmethod
    def supports_pth_moment(self, p):
        """Whether analytically computing the pth moment is supported"""
        pass

    @property
    def dim(self):
        """Dimension of the space the distribution is defined on"""
        return self._dim

    @property
    def var_param_dim(self):
        """Dimension of the variational parameter"""
        return self._var_param_dim


def _get_mu_log_sigma_pattern(dim):
    ms_pattern = PatternDict(free_default=True)
    ms_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_pattern['log_sigma'] = NumericVectorPattern(length=dim)
    return ms_pattern


class MFGaussian(ApproximationFamily):
    """A mean-field Gaussian approximation family."""
    def __init__(self, dim, seed=1):
        """Create mean field Gaussian approximation family.

        Parameters
        ----------
        dim : `int`
            dimension of the underlying parameter space
        """
        self._rs = npr.RandomState(seed)
        self._pattern = _get_mu_log_sigma_pattern(dim)
        super().__init__(dim, self._pattern.flat_length(True), True, True)

    def init_param(self):
        init_param_dict = dict(mu=np.zeros(self.dim),
                               log_sigma=2*np.ones(self.dim))
        return self._pattern.flatten(init_param_dict)

    def sample(self, var_param, n_samples, seed=None):
        my_rs = self._rs if seed is None else npr.RandomState(seed)
        param_dict = self._pattern.fold(var_param)
        return param_dict['mu'] +  np.exp(param_dict['log_sigma']) * my_rs.randn(n_samples, self.dim)

    def _entropy(self, var_param):
        param_dict = self._pattern.fold(var_param)
        return 0.5 * self.dim * (1.0 + np.log(2*np.pi)) + np.sum(param_dict['log_sigma'])

    def _kl(self, var_param0, var_param1):
        param_dict0 = self._pattern.fold(var_param0)
        param_dict1 = self._pattern.fold(var_param1)
        mean_diff = param_dict0['mu'] - param_dict1['mu']
        log_stdev_diff = param_dict0['log_sigma'] - param_dict1['log_sigma']
        return .5 * np.sum(  np.exp(2*log_stdev_diff)
                           + mean_diff**2 / np.exp(2*param_dict1['log_sigma'])
                           - 2*log_stdev_diff
                           - 1)

    def log_density(self, var_param, x):
        param_dict = self._pattern.fold(var_param)
        return mvn.logpdf(x, param_dict['mu'], np.diag(np.exp(2*param_dict['log_sigma'])))

    def mean_and_cov(self, var_param):
        param_dict = self._pattern.fold(var_param)
        return param_dict['mu'], np.diag(np.exp(2*param_dict['log_sigma']))

    def _pth_moment(self, var_param, p):
        param_dict = self._pattern.fold(var_param)
        vars = np.exp(2*param_dict['log_sigma'])
        if p == 2:
            return np.sum(vars)
        else:  # p == 4
            return 2*np.sum(vars**2) + np.sum(vars)**2

    def supports_pth_moment(self, p):
        return p in [2, 4]


class MFStudentT(ApproximationFamily):
    """A mean-field Student's t approximation family."""
    def __init__(self, dim, df, seed=1):
        if df <= 2:
            raise ValueError('df must be greater than 2')
        self._df = df
        self._rs = npr.RandomState(seed)
        self._pattern = _get_mu_log_sigma_pattern(dim)
        super().__init__(dim, self._pattern.flat_length(True), True, False)

    def init_param(self):
        init_param_dict = dict(mu=np.zeros(self.dim),
                               log_sigma=2*np.ones(self.dim))
        return self._pattern.flatten(init_param_dict)

    def sample(self, var_param, n_samples, seed=None):
        my_rs = self._rs if seed is None else npr.RandomState(seed)
        param_dict = self._pattern.fold(var_param)
        return param_dict['mu'] + np.exp(param_dict['log_sigma']) * my_rs.standard_t(self.df, size=(n_samples, self.dim))

    def entropy(self, var_param):
        # ignore terms that depend only on df
        param_dict = self._pattern.fold(var_param)
        return np.sum(param_dict['log_sigma'])

    def log_density(self, var_param, x):
        if x.ndim == 1:
            x = x[np.newaxis,:]
        param_dict = self._pattern.fold(var_param)
        return np.sum(t_dist.logpdf(x, self.df, param_dict['mu'], np.exp(param_dict['log_sigma'])), axis=-1)

    def mean_and_cov(self, var_param):
        param_dict = self._pattern.fold(var_param)
        df = self.df
        cov = df / (df - 2) * np.diag(np.exp(2*param_dict['log_sigma']))
        return param_dict['mu'], cov

    def _pth_moment(self, var_param, p):
        df = self.df
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = self._pattern.fold(var_param)
        scales = np.exp(param_dict['log_sigma'])
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(scales**2)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(scales**4) + np.sum(scales**2)**2)

    def supports_pth_moment(self, p):
        return p in [2, 4] and p < self.df

    @property
    def df(self):
        """Degrees of freedom."""
        return self._df


def _get_mu_sigma_pattern(dim):
    ms_pattern = PatternDict(free_default=True)
    ms_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_pattern['Sigma'] = PSDSymmetricMatrixPattern(size=dim)
    return ms_pattern


class MultivariateT(ApproximationFamily):
    """A full-rank multivariate t approximation family."""
    def __init__(self, dim, df, seed=1):
        if df <= 2:
            raise ValueError('df must be greater than 2')
        self._df = df
        self._rs = npr.RandomState(seed)
        self._pattern = _get_mu_sigma_pattern(dim)
        self._log_density = FlattenFunctionInput(
            lambda param_dict, x: multivariate_t_logpdf(x, param_dict['mu'], param_dict['Sigma'], df),
            patterns=self._pattern, free=True, argnums=0)
        super().__init__(dim, self._pattern.flat_length(True), True, False)

    def init_param(self):
        init_param_dict = dict(mu=np.zeros(self.dim),
                               Sigma=10*np.eye(self.dim))
        return self._pattern.flatten(init_param_dict)

    def sample(self, var_param, n_samples, seed=None):
        my_rs = self._rs if seed is None else npr.RandomState(seed)
        df = self.df
        s = np.sqrt(my_rs.chisquare(df, n_samples) / df)
        param_dict = self._pattern.fold(var_param)
        z = my_rs.randn(n_samples, self.dim)
        sqrtSigma = sqrtm(param_dict['Sigma'])
        return param_dict['mu'] + np.dot(z, sqrtSigma)/s[:,np.newaxis]

    def entropy(self, var_param):
        # ignore terms that depend only on df
        param_dict = self._pattern.fold(var_param)
        return .5*np.log(np.linalg.det(param_dict['Sigma']))

    def log_density(self, var_param, x):
        return self._log_density(var_param, x)

    def mean_and_cov(self, var_param):
        param_dict = self._pattern.fold(var_param)
        df = self.df
        return param_dict['mu'], df / (df - 2.) * param_dict['Sigma']

    def _pth_moment(self, var_param, p):
        df = self.df
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = self._pattern.fold(var_param)
        sq_scales = np.linalg.eigvalsh(param_dict['Sigma'])
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(sq_scales)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(sq_scales**2) + np.sum(sq_scales)**2)

    def supports_pth_moment(self, p):
        return p in [2, 4] and p < self.df

    @property
    def df(self):
        """Degrees of freedom."""
        return self._df


def _get_low_rank_mu_sigma_pattern(dim,M):
    ms_lr_pattern = PatternDict(free_default=True)
    ms_lr_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_lr_pattern['B'] =  NumericArrayPattern(shape=(dim,M))
    ms_lr_pattern['D'] = NumericArrayPattern(shape=(dim,))
    return ms_lr_pattern


class LowRankGaussian(ApproximationFamily):
    def __init__(self, dim, M, seed=1):
        if M > dim:
            raise ValueError('M must be smaller than original rank')
        self._rs = npr.RandomState(seed)
        self._pattern = _get_low_rank_mu_sigma_pattern(dim, M)
        self._dim = dim
        self.M = M
        self._df = 1e30
        self._log_density = FlattenFunctionInput(
        lambda x, ms_lr_dict: multivariate_t_logpdf(x, ms_lr_dict['mu'],
                                                    np.dot(ms_lr_dict['B'],
                                                           np.transpose(ms_lr_dict['B'])) +
                                                    np.power(np.diag(ms_lr_dict['D']),2), 1e30),
        patterns=self._pattern, free=True, argnums=1)
        super().__init__(dim, self._pattern.flat_length(True), True, False)


    def init_param(self):
        init_param_dict = dict(mu=np.zeros(self.dim),
                               B=1.4*np.ones(self.dim*self.M),D=np.ones(self.dim))

        return self._pattern.flatten(init_param_dict)


    def sample(self, var_param, n_samples, seed=None):
        my_rs = self._rs if seed is None else npr.RandomState(seed)
        param_dict = self._pattern.fold(var_param)
        s = np.sqrt(my_rs.chisquare(1e30, n_samples) / 1e30)
        z1 = my_rs.randn(n_samples, self.dim)
        z2 = my_rs.randn(n_samples, self.M)
        B = param_dict['B']
        D= param_dict['D']
        #D1 = np.sqrt(D)
        return param_dict['mu'] + np.dot(z2, np.transpose(B))/s[:,np.newaxis] + z1*np.transpose(D)

    def mean_and_cov(self, var_param):
        param_dict = self._pattern.fold(var_param)
        return param_dict['mu'],  np.dot(param_dict['B'],
                                                         np.transpose(param_dict['B'])) + np.power(np.diag(param_dict['D']),2)

    def entropy(self, var_param):
        param_dict = self._pattern.fold(var_param)
        Sigma = np.dot(param_dict['B'], np.transpose(param_dict['B'])) + np.power(np.diag(param_dict['D']),2)
        return .5*np.log(np.linalg.det(Sigma))

    def log_density(self, var_param, x):
        return self._log_density(var_param, x)

    def _pth_moment(self, var_param, p):
        df = self.df
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = self._pattern.fold(var_param)
        sq_scales = np.linalg.eigvalsh(param_dict['Sigma'])
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(sq_scales)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(sq_scales**2) + np.sum(sq_scales)**2)

    def supports_pth_moment(self, p):
        return p in [2, 4] and p < self.df


def _get_planar_flow_pattern(dim,num_layers):
    planar_flow_pattern = PatternDict(free_default=True)
    planar_flow_pattern['u'] = NumericArrayPattern(shape=(dim,num_layers))
    planar_flow_pattern['W'] = NumericArrayPattern(shape=(dim,num_layers))
    planar_flow_pattern['b'] =  NumericArrayPattern(shape=(1,num_layers))
    #ms_pattern['Lr'] =
    return planar_flow_pattern

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

class Norm_Flow_Planar(ApproximationFamily):
    def __init__(self, dim, num_layers, seed=1):
        self._dim = dim
        self._num_layers = num_layers
        self._seed = seed
        self._supports_entropy = True
        self._supports_kl= False
        self._rs = npr.RandomState(seed)
        self._pattern = _get_planar_flow_pattern(dim, num_layers)
        self.flow, self.flow_det = planar_flow()

    # make planar transformation functions
    flow, flow_det = planar_flow()

    def forward(self, z, var_param):
        z_current= z
        ldet_sum = np.zeros(z.shape[0])
        num_layers = self._num_layers
        pattern = self._pattern
        flow = self.flow
        flow_det = self.flow_det

        param_dict = pattern.fold(var_param)
        for l in range(num_layers):
            ul = param_dict['u'][:,l]
            wl = param_dict['W'][:,l]
            bl = param_dict['b'][:,l]
            z_current = flow(z_current, wl, bl, ul)
            ldet_sum = ldet_sum + np.log(flow_det(z_current, wl, bl, ul))

        return  z_current, ldet_sum


    def sample(self, var_param, n_samples, seed=42):
        npr.RandomState(self._seed)
        self.var_param=var_param
        self.n_samples = n_samples


        eps = self._rs.randn(n_samples, self.dim)

        zs, ldet_sum = self.forward(eps, var_param)
        return zs


    def qlogprob(self, var_param, n_samples, eps=None):
        if eps is None:
            eps = npr.randn(n_samples, self.dim)

        zs, ldet_sum = self.forward(eps, var_param)
        lls = mvn.logpdf(eps, mean=np.zeros(self.dim), cov=np.eye(self.dim)) - ldet_sum
        return lls, zs


    def qlogprob2(self, var_param, n_samples, eps=None):
        if eps is None:
            eps = npr.randn(n_samples, self.dim)

        zs, ldet_sum = self.forward(eps, var_param)
        lls = -mvn.logpdf(eps, mean=np.zeros(self.dim), cov=np.eye(self.dim)) + ldet_sum
        return lls, zs

    def log_density(self, var_param, samples):
        eps = npr.randn(self.n_samples, self.dim)
        zs, ldet_sum = self.forward(eps, self.var_param)
        lls = mvn.logpdf(eps, mean=np.zeros(self.dim), cov=np.eye(self.dim)) - ldet_sum
        return lls

    def entropy(self, var_param):
        eps = self._rs.randn(self.n_samples, self.dim)
        zs, ldet_sum = self.forward(eps, var_param)
        lls = mvn.logpdf(eps, mean=np.zeros(self.dim), cov=np.eye(self.dim)) - ldet_sum
        ldet_mean = np.mean(ldet_sum)
        return ldet_mean

    def compute_k_hat(self, n_samples, logdensity, eps=None):
        log_q, zs = self.qlogprob(self.var_param, n_samples)
        log_p = logdensity(zs)
        log_weights= log_p - log_q
        #_, paretok = psislw(log_weights)
        return log_weights

    def lnq_grid(self, var_param):
        xg, yg = np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)
        xx, yy = np.meshgrid(xg, yg)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        zs, ldets = self.forward(pts, var_param)
        lls = mvn.logpdf(pts, mean=np.zeros(self.dim), cov=np.eye(self.dim)) - ldets
        return zs[:,0].reshape(xx.shape), zs[:,1].reshape(yy.shape), lls.reshape(xx.shape)

    def get_samples_and_log_weights(self, var_param, logdensity, num_mc_samples):
        log_q, zs = self.qlogprob(var_param, num_mc_samples)
        log_p = logdensity(zs)
        log_weights= log_p - log_q
        smoothed_lw, paretok = psislw(log_weights)
        return zs, paretok, smoothed_lw


    def improve_with_psis(self, var_param, log_density, num_mc_samples,
                          true_mean, true_cov, transform=None, verbose=True):
        samples, khat, slw = self.get_samples_and_log_weights(var_param, log_density, num_mc_samples)
        if verbose:
            print('khat = {:.3g}'.format(khat))
            print()
        if transform is not None:
            samples = transform(samples)
        slw -= np.max(slw)
        wts = np.exp(slw)
        wts /= np.sum(wts)

        approx_mean = np.mean(samples, axis=0).flatten()
        approx_mean_PSIS = wts[np.newaxis, :] @ samples
        approx_mean_PSIS = approx_mean_PSIS.flatten()

        approx_cov_PSIS = np.cov(samples.T, aweights=wts, ddof=0)
        approx_cov = np.cov(samples.T, ddof=0)

        true_std = np.sqrt(np.diag(true_cov))
        approx_std = np.sqrt(np.diag(approx_cov))
        approx_std_PSIS = np.sqrt(np.diag(approx_cov_PSIS))

        results = dict(mean_error_PSIS=np.linalg.norm(true_mean - approx_mean_PSIS),
                       mean_error = np.linalg.norm(true_mean - approx_mean),
                       cov_error_2_PSIS=np.linalg.norm(true_cov - approx_cov_PSIS, ord=2),
                       cov_error_2=np.linalg.norm(true_cov - approx_cov, ord=2),
                       cov_norm_2=np.linalg.norm(true_cov, ord=2),
                       cov_error_nuc_PSIS=np.linalg.norm(true_cov - approx_cov_PSIS, ord='nuc'),
                       cov_norm_nuc=np.linalg.norm(true_cov -approx_cov, ord='nuc'),
                       std_error_PSIS=np.linalg.norm(true_std - approx_std_PSIS),
                       std_error=np.linalg.norm(true_std - approx_std),
                       rel_std_error=np.linalg.norm(approx_std / true_std - 1),
                       )

        if verbose:
            print('mean   =', approx_mean)
            print('stdevs =', approx_std)
            print('mean error            = {:.3g}'.format(results['mean_error']))
            print('stdev error            = {:.3g}'.format(results['std_error']))
            print('||cov error||_2^{{1/2}}   = {:.3g}'.format(np.sqrt(results['cov_error_2'])))
            print('mean error  PSIS           = {:.3g}'.format(results['mean_error_PSIS']))
            print('stdev error PSIS           = {:.3g}'.format(results['std_error_PSIS']))
            print('||cov error||_2^{{1/2}} PSIS  = {:.3g}'.format(np.sqrt(results['cov_error_2_PSIS'])))
            print('||true cov||_2^{{1/2}}   = {:.3g}'.format(np.sqrt(results['cov_norm_2'])))

        results['khat'] = khat
        return results, approx_mean, approx_cov

    def mean_and_cov(self, var_param):
        samples = self.sample(var_param,200000)
        return samples

    def _pth_moment(self, var_param, p):
        pass

    def supports_pth_moment(self, p):
        pass



class NVPFlow(ApproximationFamily):
    def __init__(self,d):
        self.d =d

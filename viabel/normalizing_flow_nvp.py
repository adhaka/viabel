
from autograd import value_and_grad, vector_jacobian_product, grad, elementwise_grad
from autograd.extend import primitive, defvjp

from autograd.misc.optimizers import adam, sgd


import autograd.numpy as np
import autograd.numpy.random as npr

from collections import namedtuple
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist


from  .optimization_diagnostics import autocorrelation, monte_carlo_se, monte_carlo_se2, compute_khat_iterates, gpdfit
from .psis import psislw
from misc import WeightsSlicer, kl_mvn_diag, mog_logprob


NormalizingFlowConstructor = namedtuple('NormalizingFlowModule',
                                   ['flow', 'flow_det'])

def nvp_flow(d,D):

    mask = np.zeros(D)
    mask[:d] = 1

    params = np.concatenate([np.random.randn(D*D + D),
                             np.random.randn(D*D + D)])

    def unpack_params(flow_param):
        W, b = params[:D*D], params[D*D:]
        return np.reshape(W, (D,D)), b

    def lfn(z, lparams):
        W,b= unpack_params(lparams)
        return np.tanh(np.dot(z,W) + b)


    def rfn(z, mparams):
        W,b = unpack_params(mparams)
        return np.tanh(np.dot(z,W) +b)

    def flow(z, params):
        lparams, rparams = np.split(params,2)
        return mask*z + (1-mask)*(z*np.exp(lfn(mask*z, lparams)) +
                                  rfn(mask*z, rparams))


    def flow_inv(zprime, params):
        lparams, rparams = np.split(params,2)
        a = mask*zprime
        b = (1 -mask)*(zprime -rfn(mask*zprime, rparams))*np.exp(-lfn(mask*zprime, lparams))

        return a +b

    def flow_det(z, params):
        lparams, rparams = np.split(params,2)
        diag = (1-mask)*lfn(mask*z, lparams)
        if len(z.shape) > 1:
            return np.sum(diag, axis=1)
        else:
            return np.sum(diag)

    return flow, flow_det, flow_inv, params



def black_box_nvp_flow_objective(logdensity, D, beta_schedule, num_layers=2):
    parser = WeightsSlicer()
    layers = []

    for l in range(num_layers):
        parser.add_shape("w_%d"%l, (D, D))
        parser.add_shape("b_%d"%l, (D, ))
        layers.append(("w_%d"%l, "b_%d"%l))

    # make planar transformation functions
    flow, flow_det, flow_inv, flow_params = nvp_flow(int(D/2), D)

    def forward(z, var_params, layers):
        z_current= z
        ldet_sum = np.zeros(z.shape[0])
        for l, ln in enumerate(layers):
            wl = parser.get(var_params, ln[0])
            bl = parser.get(var_params, ln[1])
            z_current = flow(z_current, var_params)
            ldet_sum = ldet_sum + np.log(flow_det(z_current, var_params))

        return  z_current, ldet_sum

    def sample(flow_params, n_samples, eps=None, seed=42):
        npr.RandomState(seed)

        if eps is None:
            eps = npr.randn(n_samples, D)

        zs, ldet_sum = forward(eps, flow_params, layers)
        return zs, ldet_sum

    def qlogprob(var_params, n_samples, eps=None):
        if eps is None:
            eps = npr.randn(n_samples, D)

        zs, ldet_sum = forward(eps, var_params, layers)
        lls = mvn.logpdf(eps, mean=np.zeros(D), cov=np.eye(D)) - ldet_sum
        print(lls.shape)

        return lls, zs
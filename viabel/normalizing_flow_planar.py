

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


def planar_flow():
    def uhat(u,w):
        what = w/np.dot(w,w)
        #what = w/np.linalg.norm(w,2)
        mwu = -1 + np.log(1 + np.exp(np.dot(w,u)))
        return u + (mwu -np.dot(w,u))*what

    def flow(z,w,u,b):
        h = np.tanh(np.dot(z,w) + b)
        return z + uhat(u, w)*h[:,None]

    def flow_det(z,w,u,b):
        x = np.dot(z,w) +b
        g = elementwise_grad(np.tanh)(np.dot(z,w) + b )[:,None]*w
        return np.abs(1 + np.dot(g, uhat(u,w)))

    return NormalizingFlowConstructor(flow, flow_det)


def black_box_norm_flow(logdensity,D, beta_schedule, num_layers=2, n_samples=100):
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
        import numpy as npp
        print('sample mean:')
        print(zs.mean(axis=0))
        print(true_mean_ncp)
        print(((zs.mean(axis=0) - true_mean_ncp)**2).sum())
        print('sample cov:')
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
        xg, yg = np.linspace(-4, 4, 50), np.linspace(-4, 4, 50)
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


def sylvester_flow(num_ortho_vectors):
    num_ortho_vectors = num_ortho_vectors
    diag_idx = np.arange(0, num_ortho_vectors)

    def dtanh(self, x):
        return 1 -  np.tanh(x) ** 2

    def flow(zk, r1,r2, q_ortho,b, sum_ldj=True, permute_z=None):
        diag_r1 = r1[:, diag_idx, diag_idx]
        diag_r2 = r2[:, diag_idx, diag_idx]
        r1hat = r1
        r2hat = r2
        qr2 = np.matmul(q_ortho, r2hat.transpose(2,1))
        qr1 = np.matmul(q_ortho, r1hat)

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = np.matmul(qr2.T, z_per) +b
        zprime  = zk +  np.matmul(qr1, np.tanh(r2qzb))

        diag_j = diag_r1 * diag_r2
        diag_j = dtanh(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        return zprime

    def flow_det(z,w,u,b):
        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j



if __name__ == '__main__':
    npr.seed(42)
    D = 2
    K = 2
    icovs = np.zeros((K, D, D))
    dets, pis = np.zeros(K), np.zeros(K)
    means = np.array([[-1.5, 1.3], [1., -1.3]])
    for k in range(len(means)):
        icovs[k, :, :] = 1/.5*np.eye(D)
        dets[k]  = 1.
        pis[k] = .5

    # log unnormalized distribution to learn
    lnpdf = lambda z, t: mog_logprob(z, means, icovs, dets, pis)

    ########################################
    # Build variational objective/gradient #
    ########################################
    objective, gradient, num_variational_params, sample_z, lnq_grid = \
        black_box_norm_flow(logdensity = lnpdf,
                                        beta_schedule = np.linspace(0., 1., 200),
                                        D=D, num_layers=3)

    #######################################################
    # set up plots + callback for monitoring optimization #
    #######################################################
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    from plots import plot_isocontours
    def make_plot_fun():
        num_plot_samps = 1000
        eps = npr.randn(num_plot_samps, D)
        def plot_q_dist(ax, params):
            zsamps, _ = sample_z(params, num_plot_samps, eps)
            print(" .... ", np.mean(zsamps, axis=0))
            xx, yy, ll = lnq_grid(params)
            ax.contour(xx, yy, np.exp(ll))
            #plt.scatter(zsamps[::2,0], zsamps[::2,1], c='grey', s=5, alpha=.5)
        return plot_q_dist

    plot_q_dist = make_plot_fun()

    # setup figure
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t, 1000)))
        if t % 10 == 0:
            plt.cla()
            # plot target
            target_distribution = lambda x : np.exp(lnpdf(x, t))
            plot_isocontours(ax, target_distribution, fill=True)
            # plot approximate distribution
            plot_q_dist(ax, params)
            ax.set_xlim((-3, 3))
            ax.set_ylim((-4, 4))
            plt.draw()
            plt.pause(1.0/30.0)

    #####################
    # Run optimization  #
    #####################
    print("Optimizing variational parameters...")
    th = .5*npr.randn(num_variational_params) - 6.

    num_objective_samps = 10
    def grad_wrap(th, t):
        return gradient(th, t, num_objective_samps)

    variational_params = adam(grad_wrap, th, step_size=.005, num_iters=10000,
                              callback=callback)


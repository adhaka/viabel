from abc import ABC, abstractmethod
from autograd import value_and_grad, vector_jacobian_product, jacobian, elementwise_grad, grad

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.core import getval
from viabel._psis import psislw
from functools import partial

__all__ = [
    'VariationalObjective',
    'StochasticVariationalObjective',
    'ExclusiveKL',
    'AlphaDivergence',
    'InclusiveKL'
]


class VariationalObjective(ABC):
    """A class representing a variational objective to minimize"""
    def __init__(self, approx, model):
        """
        Parameters
        ----------
        approx : `ApproximationFamily` object
        model : `Model` object
        """
        self._approx = approx
        self._model = model
        self._objective_and_grad = None
        self._update_objective_and_grad()


    def __call__(self, var_param):
        """Evaluate objective and its gradient.

        May produce an (un)biased estimate of both.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        """
        if self._objective_and_grad is None:
            raise RuntimeError("no objective and gradient available")
        return self._objective_and_grad(var_param)

    @abstractmethod
    def _update_objective_and_grad(self):
        """Update the objective and gradient function.

        Should be called whenever a parameter that the objective depends on
        (e.g., `approx` or `model`) is updated."""
        pass

    @property
    def approx(self):
        """The approximation family."""
        return self._approx

    @approx.setter
    def approx(self, value):
        self._approx = value
        self._update_objective_and_grad()

    @property
    def model(self):
        """The model."""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._update_objective_and_grad()


class StochasticVariationalObjective(VariationalObjective):
    """A class representing a variational objective approximated using Monte Carlo."""
    def __init__(self, approx, model, num_mc_samples):
        """
        Parameters
        ----------
        approx : `ApproximationFamily` object
        model : `Model` object
        num_mc_sample : `int`
            Number of Monte Carlo samples to use to approximate the objective.
        """
        self._num_mc_samples = num_mc_samples
        super().__init__(approx, model)

    @property
    def num_mc_samples(self):
        """Number of Monte Carlo samples to use to approximate the objective."""
        return self._num_mc_samples

    @num_mc_samples.setter
    def num_mc_samples(self, value):
        self._num_mc_samples = value
        self._update_objective_and_grad()


class ExclusiveKL(StochasticVariationalObjective):
    """Exclusive Kullback-Leibler divergence.

    Equivalent to using the canonical evidence lower bound (ELBO)
    """
    def _update_objective_and_grad(self):
        approx = self.approx
        def variational_objective(var_param):
            samples = approx.sample(var_param, self.num_mc_samples)
            if approx.supports_entropy:
                lower_bound = np.mean(self.model(samples)) + approx.entropy(var_param)
            else:
                lower_bound = np.mean(self.model(samples) - approx.log_density(samples))
            return -lower_bound
        self._objective_and_grad = value_and_grad(variational_objective)


class AlphaDivergence(StochasticVariationalObjective):
    """Log of the alpha-divergence."""
    def __init__(self, approx, model, num_mc_samples, alpha):
        """
        Parameters
        ----------
        approx : `ApproximationFamily` object
        model : `Model` object
        num_mc_sample : `int`
            Number of Monte Carlo samples to use to approximate the objective.
        alpha : `float`
        """
        self._alpha = alpha
        super().__init__(approx, model, num_mc_samples)

    @property
    def alpha(self):
        """Alpha parameter of the divergence."""
        return self._alpha

    def _update_objective_and_grad(self):
        """Provides a stochastic estimate of the variational lower bound."""
        def compute_log_weights(var_param, seed):
            samples = self.approx.sample(var_param, self.num_mc_samples, seed)
            log_weights = self.model(samples) - self.approx.log_density(var_param, samples)
            return log_weights

        log_weights_vjp = vector_jacobian_product(compute_log_weights)
        alpha = self.alpha
        # manually compute objective and gradient
        def objective_grad_and_log_norm(var_param):
            # must create a shared seed!
            seed = npr.randint(2**32)
            log_weights = compute_log_weights(var_param, seed)
            log_norm = np.max(log_weights)
            scaled_values = np.exp(log_weights - log_norm)**alpha
            obj_value = np.log(np.mean(scaled_values))/alpha + log_norm
            obj_grad = alpha*log_weights_vjp(var_param, seed, scaled_values) / scaled_values.size
            return (obj_value, obj_grad)

        self._objective_and_grad = objective_grad_and_log_norm


class FDiv(StochasticVariationalObjective):
    '''f duvergence '''

    def __init__(self, approx, model, num_mc_samples, beta, seed=42):
        self.beta = beta
        self.seed = seed
        super().__init__(approx, model, num_mc_samples)

    def _update_objective_and_grad(self):
        """Provides a stochastic estimate of the variational lower bound."""

        def objective(var_param, samples):
            logp = self.model(samples)
            logq = self.approx.log_density(var_param, samples)
            unweighted_obj = np.sum(logq - logp)
            return unweighted_obj

        def compute_log_weights(var_param):
            def compute(samples):
                log_weights = self.model(samples) - self.approx.log_density(getval(var_param), samples)
                return log_weights

            return compute

        def compute_log_weights2(var_param, seed):
            samples = self.approx.sample(var_param, self.num_mc_samples, seed)
            # log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
            log_weights = self.model(samples) - self.approx.log_density(var_param, samples)
            return log_weights

        def compute_g(var_param):
            """Provides a stochastic estimate of the variational lower bound."""
            samples = self.approx.sample(var_param, self.num_mc_samples)
            return samples

        def phi(logp, logq):
            diff = logp - logq
            norm_diff = diff - np.max(diff)
            dx = np.exp(norm_diff)
            prob = np.sign(dx[np.newaxis, ...] - dx[..., np.newaxis])
            prob = np.array(prob > 0.5, dtype=np.float32)
            wx = np.sum(prob, axis=1) / len(logp)
            wx = (1. - wx) ** self.beta
            return wx / np.sum(wx)

        def f_w(t, w):
            return np.mean(w >= t)


        # manually compute objective and gradient
        def objective_grad_and_log_norm(var_param, mode=3):
            #seed = self.seed
            seed = npr.randint(2 ** 32)

            if mode == 1:
                samples = self.approx.sample(var_param, self.num_mc_samples)
                logp = self.model(samples)
                logq = self.approx.log_density(samples, var_param)

                ograd = grad(objective, 0)
                obj_grad = ograd(var_param, samples)

                weights = phi(logp, logq)
                obj = np.sum(weights * (logq - logp))
                return obj, obj_grad

            def samples_fn(var_param):
                samples = self.approx.sample(var_param, self.num_mc_samples, seed)
                return samples

            samples = compute_g(var_param)
            compute_logw = compute_log_weights(var_param)
            particle_grads = elementwise_grad(compute_logw)(samples)
            log_weights = compute_log_weights2(var_param, seed)

            pmz_len = len(var_param)
            n_theta = pmz_len // 2

            weights = np.exp(log_weights - np.max(log_weights))
            gamma = np.array([f_w(weights[i], weights) for i in range(len(weights))]) ** self.beta
            gamma1 = gamma[:, None]
            z_gamma = np.sum(gamma1)


            # preferred mode, using autograd gradients, formulating the gradient as in eq
            if mode == 2:
                transform_grad = jacobian(samples_fn)(var_param)
                a1 = np.transpose(transform_grad, (0, 2, 1))
                b1 = np.transpose(particle_grads[:, None], (2, 1, 0))
                # grad_t1 = np.dot(a1,b1)

                l = a1.shape[1]

                grad_t2 = np.zeros((self.num_mc_samples, l))
                for i in range(l):
                    c1 = np.diag(np.dot(a1[:, i, :], b1[:, 0, :]))
                    grad_t2[:, i] = c1

                obj_grad_all = -(1. / z_gamma) * np.mean(gamma1 * grad_t2, axis=0)
                # print(obj_grad_all.shape)

            if mode == 3:
                #print(gamma1)
                obj_grad = -(1. / z_gamma) * gamma1 * particle_grads
                #print(samples.shape)
                #print(obj_grad.shape)
                #print(np.mean((samples- var_param[:n_theta]) * obj_grad * np.exp(var_param[n_theta:]), axis=0))
                obj_grad_all = np.concatenate(
                    [np.mean(obj_grad, axis=0), np.mean((samples- var_param[:n_theta]) * obj_grad * np.exp(var_param[n_theta:]), axis=0)])
                # print(obj_grad_all.shape)
                obj_value = 0

            return (0, obj_grad_all)

        self._objective_and_grad = objective_grad_and_log_norm



class FDivPSIS(FDiv):
    def _update_objective_and_grad(self):
        """Provides a stochastic estimate of the variational lower bound."""
        def compute_log_weights(var_param, seed):
            samples = self.approx.sample(var_param, self.num_mc_samples, seed)
            log_weights = self.model(samples) - self.approx.log_density(var_param, samples)
            return log_weights

        log_weights_vjp = vector_jacobian_product(compute_log_weights)
        beta = self.beta
        # manually compute objective and gradient
        def objective_grad_and_log_norm(var_param, mode=2):
            seed = self.seed

            def objective(var_param, samples):
                logp = self.model(samples)
                logq = self.approx.log_density(var_param, samples)
                unweighted_obj = np.sum(logq - logp)
                return unweighted_obj

            def compute_log_weights(var_param):
                def compute(samples):
                    log_weights = self.model(samples) - self.approx.log_density(var_param, samples)
                    return log_weights

                return compute

            def compute_log_weights2(var_param, seed):
                samples = self.approx.sample(var_param, self.num_mc_samples, seed)
                # log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
                log_weights = self.model(samples) - self.approx.log_density(var_param, samples)
                return log_weights

            def compute_g(var_param):
                """Provides a stochastic estimate of the variational lower bound."""
                samples = self.approx.sample(var_param, self.num_mc_samples)
                return samples

            def samples2(var_param):
                samples = self.approx.sample(var_param, self.num_mc_samples, seed)
                return samples

            samples = compute_g(var_param)
            compute_logw = compute_log_weights(var_param)
            particle_grads = elementwise_grad(compute_logw)(samples)
            log_weights = compute_log_weights2(var_param, seed)

            pmz_len = len(var_param)
            n_theta = pmz_len // 2

            weights = np.exp(log_weights - np.max(log_weights))

            # preferred mode, using autograd gradients, formulating the gradient as in eq
            if mode == 1:
                smoothed_lw, paretok = psislw(log_weights)
                smoothed_weights = np.exp(smoothed_lw)
                smoothed_weights1 = np.exp(smoothed_lw - np.max(smoothed_lw))
                smoothed_weights_norm1 = smoothed_weights1 / np.sum(smoothed_weights)

                transform_grad = jacobian(samples2)(var_param)
                a1 = np.transpose(transform_grad, (0, 2, 1))
                b1 = np.transpose(particle_grads[:, None], (2, 1, 0))
                # grad_t1 = np.dot(a1,b1)

                l = a1.shape[1]

                grad_t2 = np.zeros((self.num_mc_samples, l))
                for i in range(l):
                    c1 = np.diag(np.dot(a1[:, i, :], b1[:, 0, :]))
                    grad_t2[:, i] = c1

                obj_grad_all = -np.mean(smoothed_weights[:,None] * grad_t2, axis=0)
                # print(obj_grad_all.shape)

            if mode == 2:
                smoothed_lw, paretok = psislw(log_weights)
                smoothed_weights1 = np.exp(smoothed_lw)
                smoothed_weights = np.exp(smoothed_lw - np.max(smoothed_lw))
                smoothed_weights_norm = smoothed_weights / np.sum(smoothed_weights)

                obj_grad = -smoothed_weights_norm[:,None] *particle_grads
                obj_grad_all = np.concatenate([np.mean(obj_grad, axis=0), np.mean((samples- var_param[:n_theta]) * obj_grad * np.exp(var_param[n_theta:]), axis=0)])

            if mode == 3:
                log_weights=-log_weights
                smoothed_lw, paretok = psislw(log_weights)
                smoothed_weights1 = np.exp(smoothed_lw)
                smoothed_weights = np.exp(smoothed_lw - np.max(smoothed_lw))
                smoothed_weights_norm = smoothed_weights / np.sum(smoothed_weights)

                obj_grad = -smoothed_weights_norm[:,None] *particle_grads
                obj_grad_all = np.concatenate([np.mean(obj_grad, axis=0), np.mean((samples- var_param[:n_theta]) * obj_grad * np.exp(var_param[n_theta:]), axis=0)])

                #sprint()

            return (0, obj_grad_all)

        self._objective_and_grad = objective_grad_and_log_norm


class InclusiveKL(StochasticVariationalObjective):
    def __init__(self, approx, model, num_mc_samples, seed=42):
        self.seed = seed
        super().__init__(approx, model, num_mc_samples)

    def __call__(self, var_param, prev_zk):
        """Evaluate objective and its gradient.

        May produce an (un)biased estimate of both.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        """
        if self._objective_and_grad is None:
            raise RuntimeError("no objective and gradient available")
        return self._objective_and_grad(var_param, prev_zk)


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

    def _update_objective_and_grad(self):
        approx = self.approx
        num_mc_samples =self.num_mc_samples
        seed=self.seed
        model = self.model

        def logdensity_q(var_param):
            return partial(approx.log_density, var_param=var_param)

        def logdensity_q_grad(var_param):
            score_gradient = elementwise_grad(logdensity_q)
            return score_gradient

        def dlogq_dmu_t(x, qmu, qlogsigma, nu):
            e = (x.flatten() - qmu.flatten())
            qsigma = np.exp(qlogsigma)
            grad = ((nu + 1) * e) / (nu * qsigma ** 2 + (e ** 2))
            return -grad

        def dlogq_dsigma_t(x, qmu, qlogsigma, nu):
            e = (x.flatten() - qmu.flatten())
            qsigma = np.exp(qlogsigma)
            qsigma2 = qsigma ** 2
            e2 = np.square(e)
            dlogpdf_dvar = nu * (e2 - qsigma2) / (2 * qsigma2 * (qsigma2 * nu + e2))
            dlogpdf_dvar = dlogpdf_dvar * (2 * qsigma*2)
            grad = -dlogpdf_dvar
            return grad

        def dlogq_dmu(x, qmu, qlogsigma):
            qsigma = np.exp(qlogsigma)
            grad = (x.flatten() - qmu.flatten()) / qsigma.flatten() ** 2
            return -grad

        def dlogq_dsigma(x, qmu, qlogsigma):
            qsigma = np.exp(qlogsigma)
            e = x.flatten() - qmu.flatten()
            s_4 = 1.0 / qsigma ** 4
            grad = -(((qmu.flatten() + qsigma.flatten() - x.flatten()) *
                      (-qmu.flatten() + qsigma.flatten() + x.flatten())) / qsigma.flatten() ** 2)
            # return -0.5/qsigma* + 0.5*s_4*e**2
            return -grad


        seed = npr.randint(2**32)
        #log_weights = compute_log_weights(var_param, seed

        def objective_grad_and_log_norm_t(var_param, prev_zk):
            k = prev_zk.size
            seed = npr.randint(2**32)
            #log_weights = compute_log_weights(var_param, seed)
            samples = approx.sample(var_param,num_mc_samples , seed)
            samples_all = np.concatenate((samples, prev_zk))
            log_weights = model(samples) - approx.log_density(var_param, samples)
            weight_zk = model(prev_zk) - approx.log_density(var_param, prev_zk)
            weight_zk = np.array(weight_zk)

            if log_weights.ndim ==2:
                log_weights = log_weights[0]
            log_weights_all = np.concatenate((log_weights, weight_zk))
            weights_all = np.exp(log_weights_all)
            weights_snis = np.nan_to_num(weights_all) /np.nansum(weights_all)
            weights_snis1=np.nan_to_num(weights_snis)

            idx_sample = np.random.choice(num_mc_samples +1, 1, p=weights_snis1.flatten())
            weight_sample = samples_all[idx_sample]
            obj_grad = np.concatenate([dlogq_dmu_t(weight_sample, var_param[:k], var_param[k:], approx.df),
                        dlogq_dsigma_t(weight_sample, var_param[:k], var_param[k:], approx.df)])
            obj_val = np.mean(weights_all*log_weights_all)
            return (obj_val, obj_grad, weight_sample)

        def objective_grad_and_log_norm(var_param, prev_zk):
            k = prev_zk.size
            samples = approx.sample(var_param, num_mc_samples, seed)
            samples_all = np.concatenate((samples, prev_zk))
            log_weights = model(samples) - approx.log_density(var_param, samples)
            weight_zk = model(prev_zk) - approx.log_density(var_param, prev_zk)
            weight_zk = np.array(weight_zk)

            if log_weights.ndim == 2:
                log_weights = log_weights[0]

            log_weights_all = np.concatenate((log_weights, weight_zk))
            weights_all = np.exp(log_weights_all)
            weights_snis = weights_all / np.sum(weights_all)
            idx_sample = np.random.choice(num_mc_samples + 1, 1, p=weights_snis.flatten())
            weight_sample = samples_all[idx_sample]

            #obj_grad = logdensity_q_grad(var_param)
            obj_grad = np.concatenate([dlogq_dmu(weight_sample, var_param[:k], var_param[k:]),
                                       dlogq_dsigma(weight_sample, var_param[:k], var_param[k:])])
            obj_val = np.mean(weights_all * log_weights_all)
            return (obj_val, obj_grad, weight_sample)


        self._objective_and_grad = objective_grad_and_log_norm_t
        #self._objective_and_grad = objective_grad_and_log_norm


class Per_KL(StochasticVariationalObjective):
    def _update_objective_and_grad(self):
        def variational_objective(var_param):
            def compute_log_weights(var_param, seed):
                samples = self.approx.sample(var_param, self.num_mc_samples, seed)
                log_weights = self.model(samples) - self.approx.log_density(var_param, samples)
                return log_weights
            lamda, vo = var_param[:-1], var_param[-1]
            seed = npr.randint(2**32)
            log_weights = compute_log_weights(lamda, seed)

            sum = log_weights + vo
            obj1 = sum
            obj2 = (sum ** 2) / 2
            obj3 = (sum ** 3) / 6
            obj = obj1 + obj2 + obj3
            lower_bound = np.mean(obj * np.exp(var_param[-1]))
            return -lower_bound

        obj_grad_vjp = vector_jacobian_product(variational_objective)
        def objective_grad_and_log_norm(var_param, fix_vo=True):
            obj_value = variational_objective(var_param)
            obj_grad = obj_grad_vjp(var_param, obj_value)
            if fix_vo:
                obj_grad[-1] = 0.

            return obj_value, obj_grad
        self._objective_and_grad = value_and_grad(variational_objective)
        #self._objective_and_grad = objective_grad_and_log_norm



class InclusiveKL_PSIS(InclusiveKL):

    def _update_objective_and_grad(self):
        approx = self.approx
        num_mc_samples =self.num_mc_samples
        seed=self.seed
        model = self.model

        def logdensity_q(var_param):
            return partial(approx.log_density, var_param=var_param)

        def logdensity_q_grad(var_param):
            score_gradient = elementwise_grad(logdensity_q)
            return score_gradient

        def dlogq_dmu_t(x, qmu, qlogsigma, nu):
            e = (x.flatten() - qmu.flatten())
            qsigma = np.exp(qlogsigma)
            grad = ((nu + 1) * e) / (nu * qsigma ** 2 + (e ** 2))
            return -grad

        def dlogq_dsigma_t(x, qmu, qlogsigma, nu):
            e = (x.flatten() - qmu.flatten())
            qsigma = np.exp(qlogsigma)
            qsigma2 = qsigma ** 2
            e2 = np.square(e)
            dlogpdf_dvar = nu * (e2 - qsigma2) / (2 * qsigma2 * (qsigma2 * nu + e2))
            dlogpdf_dvar = dlogpdf_dvar * (2 * qsigma*2)
            grad = -dlogpdf_dvar
            return grad

        def dlogq_dmu(x, qmu, qlogsigma):
            qsigma = np.exp(qlogsigma)
            grad = (x.flatten() - qmu.flatten()) / qsigma.flatten() ** 2
            return -grad

        def dlogq_dsigma(x, qmu, qlogsigma):
            qsigma = np.exp(qlogsigma)
            e = x.flatten() - qmu.flatten()
            s_4 = 1.0 / qsigma ** 4
            grad = -(((qmu.flatten() + qsigma.flatten() - x.flatten()) *
                      (-qmu.flatten() + qsigma.flatten() + x.flatten())) / qsigma.flatten() ** 2)
            # return -0.5/qsigma* + 0.5*s_4*e**2
            return -grad


        seed = npr.randint(2**32)
        #log_weights = compute_log_weights(var_param, seed

        def objective_grad_and_log_norm_t(var_param, prev_zk, mode=2):
            if mode == 1:
                k = prev_zk.size
                seed = npr.randint(2**32)
                #log_weights = compute_log_weights(var_param, seed)
                samples = approx.sample(var_param,num_mc_samples , seed)
                samples_all = np.concatenate((samples, prev_zk))
                log_weights = model(samples) - approx.log_density(var_param, samples)
                weight_zk = model(prev_zk) - approx.log_density(var_param, prev_zk)
                weight_zk = np.array(weight_zk)

                if log_weights.ndim ==2:
                    log_weights = log_weights[0]
                log_weights_all = np.concatenate((log_weights, weight_zk))
                weights_all = np.exp(log_weights_all)
                weights_snis = weights_all /np.sum(weights_all)
                idx_sample = np.random.choice(num_mc_samples +1, 1, p=weights_snis.flatten())
                weight_sample = samples_all[idx_sample]
                obj_grad = np.concatenate([dlogq_dmu_t(samples_all, var_param[:k], var_param[k:], approx.df),
                            dlogq_dsigma_t(samples_all, var_param[:k], var_param[k:], approx.df)])
                obj_val = np.mean(weights_all*log_weights_all)
                return (obj_val, obj_grad, weight_sample)
            elif mode == 2:
                k = prev_zk.size
                seed = npr.randint(2 ** 32)
                # log_weights = compute_log_weights(var_param, seed)
                samples = approx.sample(var_param, num_mc_samples, seed)
                samples_all = np.concatenate((samples, prev_zk))
                log_weights = model(samples) - approx.log_density(var_param, samples)
                weight_zk = model(prev_zk) - approx.log_density(var_param, prev_zk)
                weight_zk = np.array(weight_zk)

                if log_weights.ndim == 2:
                    log_weights = log_weights[0]
                log_weights_all = np.concatenate((log_weights, weight_zk))
                smoothed_lw, paretok = psislw(log_weights_all)
                weights_all = np.exp(smoothed_lw)
                weights_snis = weights_all / np.sum(weights_all)
                idx_sample = np.random.choice(num_mc_samples + 1, 1, p=weights_snis.flatten())
                weight_sample = samples_all[idx_sample]
                obj_grad = np.concatenate([dlogq_dmu_t(samples_all, var_param[:k], var_param[k:], approx.df),
                                           dlogq_dsigma_t(samples_all, var_param[:k], var_param[k:], approx.df)])
                obj_val = np.mean(weights_all * log_weights_all)
                return (obj_val, obj_grad, weight_sample)


        #def objective_grad_and_log_norm_rb(var_param, prev_zk):
        #    seed = npr.randint(2 ** 32)
        #    # log_weights = compute_log_weights(var_param, seed)
        #    samples = var_family.sample(var_param, n_samples, seed)
        #    samples_all = np.concatenate((samples, prev_zk))
        #    log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        #    weight_zk = logdensity(prev_zk) - var_family.logdensity(prev_zk, var_param)
        #    print(weight_zk.shape)
        #    log_weights_all = np.concatenate((log_weights, weight_zk))
        #    weights_all = np.exp(log_weights_all)
        #    weights_snis = weights_all / np.sum(weights_all)
        #    print(weights_snis)
        #    idx_sample = np.random.choice(n_samples + 1, 1, p=weights_snis.flatten())
        #    weight_sample = samples_all[idx_sample]
        #    obj_grad = logdensity_q_grad(var_param)
        #    var_grad = np.concatenate([dlogq_dmu_vec(samples_all, var_param[:k], var_param[k:]),
        #                               dlogq_dsigma_vec(samples_all, var_param[:k], var_param[k:])], axis=1)

        #    obj_val = np.mean(weights_all * log_weights_all)
        #    obj_grad = weights_snis @ var_grad
        #    return (obj_val, obj_grad, weight_sample)

        self._objective_and_grad = objective_grad_and_log_norm_t
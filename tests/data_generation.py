#!/usr/bin/env python3
import numpy as np
import matplotlib
import seaborn as sns
import  matplotlib.pyplot as plt



def uniform_correlation(scale, rho, show_covar_pdf=True):
    cov = np.diag(scale**2)
    D = cov.shape[1]
    for i in np.arange(D):
        for j in np.arange(i + 1, D):
            cov[i, j] = scale[i] * scale[j] * rho
    # populate lower triangular matrix
    return cov + cov.T - np.diag(cov.diagonal())



def uniform_correlation2(scale, rho):
    pass

def banded_precision(D, rho, seed=42):
    np.random.seed(seed=seed)
    K_mat = np.zeros((D, D))

    for i in range(K_mat.shape[0]):
        for j in range(K_mat.shape[1]):
            K_mat[i, j] = rho ** np.abs(i - j)

    K_mat_precision = np.linalg.inv(K_mat)
    return K_mat_precision

def banded_correlation(D, rho, seed=42):
    np.random.seed(seed=seed)
    K_mat = np.zeros((D,D))

    for i in range(K_mat.shape[0]):
        for j in range(K_mat.shape[1]):
            K_mat[i,j] = rho**np.abs(i-j)

    return K_mat


def low_rank_covariance():
    pass

def generate_data_linear(N, D, alpha=1., cov_structure='uniform', rho=0.5):
    X_mean = np.zeros(D)
    alpha_I = alpha*np.eye(D)

    noise_variance=0.6
    scale = np.ones(D)
    if cov_structure == 'uniform':
        cov = uniform_correlation(scale, rho)
    elif cov_structure== 'banded':
        cov = banded_correlation(D, rho)
    elif cov_structure == 'banded_precision':
        cov = banded_correlation(D, rho)

    X = np.random.multivariate_normal(X_mean, cov, (N,))
    beta = np.random.multivariate_normal(X_mean, alpha_I, (1,)).T
    y_mean = X @ beta
    Y = y_mean + np.random.multivariate_normal(np.array([0.]),
                                               np.eye(1) * noise_variance, (N,))
    regression_data = {}
    regression_data['X'] = X
    regression_data['Y'] = Y
    regression_data['W'] = beta
    regression_data['cov'] = cov

    return regression_data


def generate_data_logistic(N, D, alpha=1., cov_structure='uniform', rho=0.5):

    reg_data = generate_data_linear(N,D, alpha=1., cov_structure='uniform', rho=0.5)



if __name__ == '__main__':
    N=200
    D=20

    lin_reg_data= generate_data_linear(N,D)
    plt.imshow(lin_reg_data['cov'])
    plt.colorbar()
    plt.savefig('uniform_cov.pdf')
    plt.show()

    plt.scatter(lin_reg_data['X'][:,0], lin_reg_data['Y'])
    plt.show()

    lin_reg_data= generate_data_linear(N,D, cov_structure='banded', rho=0.9)
    plt.imshow(lin_reg_data['cov'])
    plt.colorbar()
    plt.savefig('banded_cov1.pdf')
    plt.show()


    plt.scatter(lin_reg_data['X'][:,0], lin_reg_data['Y'] )
    plt.show()

    lin_reg_data= generate_data_linear(N,D, cov_structure='banded_precision')
    plt.imshow(lin_reg_data['cov'])
    plt.colorbar()
    plt.savefig('banded_cov2.pdf')
    plt.show()

    plt.scatter(lin_reg_data['X'][:,0],lin_reg_data['Y'])
    plt.show()



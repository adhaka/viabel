import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
import pyprind
import autograd.numpy as np


def plot_isocontours(ax, func, xlim=[-5, 5], ylim=[-5, 5], numticks=501,
                     fill=False, vectorized=True, colors=None, levels=None):
    x = np.linspace(*xlim, num=numticks)
    y = np.linspace(*ylim, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z)
    ax.set_yticks([])
    ax.set_xticks([])


def plot_function(ax, func, xlim=[-5, 5], ylim=[-5, 5], numticks=501,
                  vectorized=True, isocontours=False,
                  cmap='Blues', alpha=1.):
    X, Y, Z = eval_fun_2d(func, xlim, ylim, numticks, vectorized)
    return ax.imshow(Z, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                     cmap=cmap, alpha=alpha, origin='lower')


def eval_fun_2d(func, xlim, ylim, numticks, vectorized):
    import numpy as np
    x    = np.linspace(*xlim, num=numticks)
    y    = np.linspace(*ylim, num=numticks)
    X, Y = np.meshgrid(x, y)
    pts  = np.column_stack([X.ravel(), Y.ravel()])
    if vectorized:
        Z = func(pts).reshape(X.shape)
    else:
        Z = np.array([func(xy)
                      for xy in pyprind.prog_bar(pts)]).reshape(X.shape)
    return X, Y, Z
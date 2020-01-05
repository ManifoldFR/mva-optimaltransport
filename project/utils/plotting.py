import numpy as np
from numpy import ndarray

from matplotlib import cm, pyplot as plt

extent_ = [0, 1, 0, 1]


def plot_domain(a: ndarray, cax=None, cmap=cm.Greys, extent=extent_,
                zorder=4, interpolation='none', **kwargs):
    """Plot the domain in black & white"""
    if cax is None:
        cax = plt.gca()
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.72
    kwargs['cmap'] = cmap
    kwargs['zorder'] = zorder
    kwargs['interpolation'] = interpolation
    return cax.imshow(a, origin='lower', extent=extent,
                      **kwargs)


def plot_measure(a: ndarray, cax: plt.Axes=None, cmap=cm.viridis, extent=extent_,
                 zorder=4, **kwargs):
    if cax is None:
        cax = plt.gca()
    kwargs['cmap'] = cmap
    return cax.imshow(a, origin='lower', extent=extent, interpolation='none', **kwargs)
    #return cax.contourf(a, extent=extent, levels=40, **kwargs)


def send_zero_transparent(a: ndarray):
    res = np.zeros(a.shape + (4,))  # define image
    res[..., 3] = a / a.max()
    return res


def hilbert_plot(hmetric: ndarray, thresh=None, sigma: float=1., title=None):
    fig = plt.figure(figsize=(4, 4))

    plt.plot(hmetric, lw=1.2)
    plt.yscale("log")
    if title is None:
        title = ("Hilbert metric $d_{\mathcal H}$ ($\\eta=%.2e$, $\\sigma=%.3f$)"
                 % (thresh, sigma))
    plt.title(title)
    plt.xlabel("Iteration no. $n$")
    fig.tight_layout()
    return fig
    

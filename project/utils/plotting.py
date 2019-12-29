import numpy as np
from numpy import ndarray

from matplotlib import cm, pyplot as plt

extent_ = [0, 1, 0, 1]


def plot_domain(a: ndarray, cax=None, cmap=cm.Greys, extent=extent_,
                zorder=4, interpolation='none', **kwargs):
    """Plot the domain in black & white"""
    if cax is None:
        cax = plt.gca()
    if not 'alpha' in kwargs:
        kwargs['alpha'] = 0.62
    kwargs['cmap'] = cmap
    kwargs['zorder'] = zorder
    kwargs['interpolation'] = interpolation
    return cax.imshow(a, origin='lower', extent=extent,
                      **kwargs)


def plot_measure(a: ndarray, cax=None, cmap=cm.Blues, extent=extent_,
                 zorder=4, interpolation='none', **kwargs):
    if cax is None:
        cax = plt.gca()
    kwargs['cmap'] = cmap
    return cax.imshow(a, origin='lower', extent=extent,
                      interpolation='none', **kwargs)


def send_zero_transparent(a: ndarray):
    res = np.zeros(a.shape + (4,))  # define image
    res[..., 3] = a/a.max()
    return res

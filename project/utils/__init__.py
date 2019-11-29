import numpy as np
from numpy import ndarray

from matplotlib import cm, pyplot as plt

extent_ = [0, 1, 0, 1]


def plot_domain(mask, cax=None, cmap=cm.Greys,
                extent=extent_, **kwargs):
    """Plot the domain in black & white"""
    if cax is None:
        cax = plt.gca()
    if not 'alpha' in kwargs:
        kwargs['alpha'] = 0.62
    kwargs['cmap'] = cmap
    return cax.imshow(mask, origin='lower', extent=extent,
               interpolation='none', zorder=5, **kwargs)

def plot_measure(a: ndarray, cax=None, cmap=cm.Blues,
                 extent=extent_, **kwargs):
    if cax is None:
        cax = plt.gca()
    kwargs['cmap'] = cmap
    return cax.imshow(a, origin='lower', extent=extent,
               interpolation='none', **kwargs)

def send_zero_transparent(a: ndarray):
    res = np.zeros(a.shape + (4,))
    res[..., 3] = a/a.max()
    return res

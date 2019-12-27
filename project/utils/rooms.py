"""
A few predefined room masks.
"""
import numpy as np
from numpy import ndarray

def room1(xg: ndarray, yg: ndarray):
    mask = (np.abs(yg-0.66) <= 0.06) & (xg <= 0.7)
    # Large block
    mask[:] = mask | ((np.abs(xg - 0.6) < 0.1) & (np.abs(yg - 0.56) < 0.1))

    mask[:] = mask | (np.abs(yg - 0.3) <= 0.04) & ((xg <= 0.10) | (xg >= 0.24))

    mask[:] = mask | ((xg <= 0.015) | (xg >= 0.985))
    mask[:] = mask | ((yg >= 0.985))
    mask[:] = mask | ((yg <= 0.015) & ((xg <= 0.64) | (xg >= 0.8)))
    return mask

def room1_bis(xg: ndarray, yg: ndarray):
    mask = (np.abs(yg-0.66) <= 0.06) & (xg <= 0.7)

    mask[:] = mask | (np.abs(yg - 0.3) <= 0.04) & ((xg <= 0.10) | (xg >= 0.24))

    mask[:] = mask | ((xg <= 0.015) | (xg >= 0.985))
    mask[:] = mask | ((yg >= 0.985))
    mask[:] = mask | ((yg <= 0.015) & ((xg <= 0.64) | (xg >= 0.8)))
    return mask


def room2(xg: ndarray, yg: ndarray):
    """
    Set up for y_max = 2.0
    """
    mask = np.zeros_like(xg, dtype=bool)
    print("Mask shape:", mask.shape)
    # Large block
    mask[:] = mask | ((xg <= 0.46) & (np.abs(yg-.9) < .5))
    mask[:] = mask | ((xg >= 0.54) & (np.abs(yg-.9) < .5))
    mask[:] = mask | (((xg <= .48) | (xg >= .52)) & (np.abs(yg-0.5) < .018))
    #mask[:] = mask & ~(((xg >= 0.53) & (xg <= .93) & (np.abs(yg-.39) <= .016)))
    #mask[:] = mask & ~((xg >= .9) & (xg <= .93) & (yg <= .39))

    mask[:] = mask | ((xg <= .3) & (yg <= 1.0))

    mask[:] = mask | ((xg <= 0.015) | (xg >= 0.985))
    mask[:] = mask | ((yg >= 1.985))
    mask[:] = mask | ((yg <= 0.015))
    return mask

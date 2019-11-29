import numpy as np
from numpy import ndarray


def room2(nx: int, xg: ndarray, yg: ndarray):
    mask = np.zeros((nx, nx), dtype=bool)
    # Large block
    mask[:] = mask | ((xg <= 0.45) & (np.abs(yg-.42) < .2))
    mask[:] = mask | ((xg >= 0.55) & (np.abs(yg-.42) < .2))

    mask[:] = mask | ((xg <= .3) & (yg <= .5))

    mask[:] = mask | ((xg <= 0.015) | (xg >= 0.985))
    mask[:] = mask | ((yg >= 0.985))
    mask[:] = mask | ((yg <= 0.015))
    return mask

"""
A few predefined room masks.
"""
import numpy as np
from numpy import ndarray


def room1(xg: ndarray, yg: ndarray) -> ndarray:
    """Setup for `[0,1]*[0,1]`."""
    mask = np.zeros_like(xg, dtype=bool)
    obst = np.abs(xg - 0.6) < 0.06
    obst[:] = obst & (np.cos(4*np.pi*yg) > -0.992)
    mask[:] = mask | obst
    return mask


def room2(xg: ndarray, yg: ndarray) -> ndarray:
    """
    Set up for y_max = 2.0
    """
    mask = np.zeros_like(xg, dtype=bool)
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


def room3(xg: ndarray, yg: ndarray) -> ndarray:
    mask = (np.abs(yg-0.66) <= 0.06) & (xg <= 0.7)
    # Large block
    mask[:] = mask | ((np.abs(xg - 0.6) < 0.1) & (np.abs(yg - 0.6) < 0.1))
    mask[:] = mask | (np.abs(yg - 0.3) <= 0.04) & ((xg <= 0.10) | (xg >= 0.24))
    mask[:] = mask | ((xg <= 0.015) | (xg >= 0.985))
    mask[:] = mask | ((yg >= 0.985))
    mask[:] = mask | ((yg <= 0.015) & ((xg <= 0.64) | (xg >= 0.8)))
    return mask


def room3_bis(xg: ndarray, yg: ndarray) -> ndarray:
    mask = (np.abs(yg-0.66) <= 0.06) & (xg <= 0.7)
    mask[:] = mask | (np.abs(yg - 0.3) <= 0.04) & ((xg <= 0.10) | (xg >= 0.24))
    mask[:] = mask | ((xg <= 0.015) | (xg >= 0.985))
    mask[:] = mask | ((yg >= 0.985))
    mask[:] = mask | ((yg <= 0.015) & ((xg <= 0.64) | (xg >= 0.8)))
    return mask


def setup1(xg: ndarray, yg: ndarray):
    rho_0 = (np.abs(xg - 0.16) <= 0.12) & (np.abs(yg - .5) <= 0.16)
    #rho_0 = (np.abs(xg - 0.16) <= 0.12) & (np.abs(yg - .18) <= 0.16)
    #rho_0[:] = rho_0 | (np.abs(xg - 0.16) <= 0.12) & (np.abs(yg - .82) <= 0.16)
    rho_0 = rho_0.astype(np.float64)
    rho_0 /= rho_0.sum()  # normalize the density
    mask = room1(xg, yg)
    exit_mask = (np.abs(xg - .88) <= 0.05) & (np.abs(yg - .5) <= 0.2)
    return rho_0, mask, exit_mask


def setup2(xg: ndarray, yg: ndarray):
    rho_0 = (np.abs(xg - 0.5) <= 0.22) & (np.abs(yg - 1.7) <= 0.2)
    rho_0 = rho_0.astype(np.float64)
    rho_0 /= rho_0.sum()  # normalize the density
    mask = room2(xg, yg)
    exit_mask = (np.abs(xg - .84) <= 0.12) & (np.abs(yg - .41) <= 0.24)
    return rho_0, mask, exit_mask


def setup3(xg: ndarray, yg: ndarray):
    rho_0 = (np.abs(xg - 0.24) <= 0.14) & (np.abs(yg - .86) <= 0.1)
    rho_0 = rho_0.astype(np.float64)
    rho_0 /= rho_0.sum()  # normalize the density
    mask = room3(xg, yg)
    exit_mask = (np.abs(xg - .8) <= 0.08) & (np.abs(yg - .14) <= 0.08)
    return rho_0, mask, exit_mask

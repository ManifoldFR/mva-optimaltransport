#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.rooms import room1
from utils import plot_domain, plot_measure, send_zero_transparent
import time
from fastsweeper.sweep import init_grid, update_grid, fast_sweep


plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['figure.dpi'] = 80


# # Domain setup

nx = 101
xar = np.linspace(0, 1, nx)
xg, yg = np.meshgrid(xar, xar)

extent = [0, 1, 0, 1]


# Obstacle domain
mask = room1(xg, yg)
obstacle_idx = np.argwhere(mask)
domain_img = send_zero_transparent(mask)

# Exit (target)
exit_mask = (yg < 0.01)

exit_layer = np.zeros((nx, nx, 4))
exit_layer[exit_mask, 0] = 1.
exit_layer[exit_mask, 3] = 1.

# fig = plt.figure()
# plot_domain(domain_img)
# plot_domain(exit_layer)
# plt.axis('off')



def progressive_grid_update():
    dx = 1. / (nx-1)

    C = np.empty((nx, nx))
    speed_field = np.ones_like(C)
    speed_field[mask] = 1e5
    init_grid(C, exit_mask, init_value=2)


    plot_domain(domain_img)
    im_ = plot_measure(C)
    plt.colorbar(im_)
    plt.axis('off');

    from IPython import display


    for k in range(40):
        C_img = C.copy()
        C_img[mask] = None
        im_ = plot_measure(C_img)
        plt.axis('off');
        plt.title("Iterate %d" % k)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(0.1)
        
        update_grid(speed_field, dx, C, False, False)
        update_grid(speed_field, dx, C, True, False)
        update_grid(speed_field, dx, C, True, True)
        update_grid(speed_field, dx, C, False, True)


# ## Cython sweep

dx = 1. / (nx-1)

C = np.empty((nx, nx))
speed_field = np.ones_like(C)
speed_field[mask] = 1e3
fast_sweep(speed_field, dx, exit_mask, 60, C)


C_filtered = np.ma.array(C, mask=mask)

fig = plt.figure(dpi=90)
plot_domain(domain_img, extent=None, cmap=cm.binary)
ct = plt.contour(C_filtered, levels=40)
plt.clabel(ct, fontsize='small')
plt.colorbar();
plt.axis('off');
plt.title("Distance map to exit")
plt.tight_layout()
plt.show()

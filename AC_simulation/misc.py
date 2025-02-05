
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:12:17 2025

@author: Erik Ewald
"""


import numpy as np
import matplotlib.pyplot as plt

def plot_voxels(obj):
    fig_3d = plt.figure()
    ax3d = fig_3d.add_subplot(projection='3d')
    if isinstance(obj, (list, tuple)):
        for p in obj:
            ax3d.voxels(p)
    else:
        ax3d.voxels(obj)
        ax3d.set_xlim3d([0, obj.shape[0]])
        ax3d.set_ylim3d([0, obj.shape[1]])
        ax3d.set_zlim3d([0, obj.shape[2]])
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        
def Rv_t_s(n, FWHM):
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    return 1 - sigma/(n * np.sqrt(2 * np.pi))
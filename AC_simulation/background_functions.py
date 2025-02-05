#!/home/eewa/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 00:12:52 2025

@author: Erik Ewald
"""

import numpy as np
import numba as nb
import random
import scipy.ndimage as ndi

class bkg_functions:

    def step(x, y, z, lim_x = 0, lim_y = 0, lim_z = 0, high_value = 1, low_value = 0):
        if not lim_x: lim_x = x.shape[0]/2
        if not lim_y: lim_y = y.shape[0]/2
        if not lim_z: lim_z = z.shape[0]/2
        return np.where((x > lim_y) & (y > lim_y) & (z > lim_z), high_value, low_value)
    
    def distance(x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)
    

    def linear_x(x,y,z):
        return x
    

    def linear_y(x,y,z):
        return y


    def linear_z(x,y,z):
        return z
        
    @classmethod
    def get_random_fn(cls):
        fn_L = [cls.step, cls.distance, cls.linear_x, cls.linear_y, cls.linear_z]
        return random.choice(fn_L)
    @classmethod
    def get_from_idx(cls, idx):
        fn_L = [cls.step, cls.distance, cls.linear_x, cls.linear_y, cls.linear_z]
        return fn_L[idx]

def make_background(mask_builder_obj, function, A_C, factor, norm = True):
    
    bkg_mask = np.ones(mask_builder_obj.shape) - mask_builder_obj.mask
    if function == None:
        function = lambda x,y,z: np.int8(1)
    bkg = function(mask_builder_obj.xx, mask_builder_obj.yy, mask_builder_obj.zz).astype(np.float64)
    
    if norm:
        bkg /= np.mean(bkg)
    
    return bkg * bkg_mask * A_C * factor
    
def calculate_background(image, mask, dist, thickness):
    inner_mask = ndi.binary_dilation(mask, iterations=dist)
    outer_mask = ndi.binary_dilation(inner_mask, iterations=thickness)
    shell_mask = outer_mask.astype(np.int8) - inner_mask.astype(np.int8)
    
    com = ndi.center_of_mass(shell_mask)
    # print(com)
    
    x_shell = np.zeros_like(shell_mask)
    x_shell[:, int(com[0]):,:] = shell_mask[:, int(com[0]):,:]
    nx_shell = shell_mask - x_shell
    
    y_shell = np.zeros_like(shell_mask)
    y_shell[int(com[1]):,:,:] = shell_mask[int(com[1]):,:,:]
    ny_shell = shell_mask - y_shell
    
    z_shell = np.zeros_like(shell_mask)
    z_shell[:,:, int(com[2]):] = shell_mask[:,:, int(com[2]):] 
    nz_shell = shell_mask - z_shell
    
    shell_list = [x_shell, nx_shell, y_shell, ny_shell, z_shell, nz_shell]
    shell_labels = ["b_x", "b_nx", "b_y", "b_ny", "b_z", "b_nz"]
    
    # fig, axes = plt.subplots(1, 3, figsize = [9,3])
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace = 0.25, hspace = 0.25)
    # _mask_size = mask.shape[0] // 2
    # axes[0].imshow(inner_mask[_mask_size // 2,:,:])
    # axes[0].set_ylabel("x")
    # axes[0].set_xlabel("z")
    # axes[1].imshow(outer_mask[:,_mask_size // 2,:])
    # axes[1].set_xlabel("z")
    # axes[1].set_ylabel("y")
    # axes[2].imshow(shell_mask[:,:, _mask_size // 2])
    # axes[2].set_xlabel("x")
    # axes[2].set_ylabel("y")
    
    # elp = z_shell
    # fig, axes = plt.subplots(1, 3, figsize = [9,3])
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace = 0.25, hspace = 0.25)
    # axes[0].imshow(elp[_mask_size // 2,:,:])
    # axes[0].set_ylabel("x")
    # axes[0].set_xlabel("z")
    # axes[1].imshow(elp[:,_mask_size // 2,:])
    # axes[1].set_xlabel("z")
    # axes[1].set_ylabel("y")
    # axes[2].imshow(elp[:,:, _mask_size // 2])
    # axes[2].set_xlabel("x")
    # axes[2].set_ylabel("y")
    
    # plot_voxels(x_shell)
    # plot_voxels(shell_mask - x_shell)
    # plot_voxels(y_shell)
    # plot_voxels(shell_mask - y_shell)
    # plot_voxels(z_shell)
    # plot_voxels(shell_mask - z_shell)
    
    return_dict = {} 
    for msk, key in zip(shell_list, shell_labels):

        return_dict[key] = np.mean(image * msk)

    return return_dict


size = 64
xx, yy, zz = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
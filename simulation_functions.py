#!/home/eewa/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:29:14 2024

@author: Erik Ewald
"""

import os
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.signal as signal
import sys
import numba as nb
from numpy.random import rand
import pandas as pd
import random

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


def make_ellipsoid(mesh_L, a, b, c, r, shift = [0,0,0]):
    xx, yy, zz = mesh_L
    px, py, pz = len(xx)/2, len(yy)/2, len(zz)/2
    xx, yy, zz = xx - px - shift[0], yy - py - shift[1], zz - pz - shift[2]
    return np.where((xx / a)**2 + (yy / b)**2 + (zz / c)**2 <= r**2, np.int8(1), np.int8(0))


class mask_builder:
    def __init__(self, shape, voxel_res):
        if isinstance(shape, int):
            self.shape = [shape]*3
        elif isinstance(shape, (list, np.ndarray)):
            self.shape = shape
            if len(shape) != 3:
                raise(IndexError(f"List or array must be of length 3. Current: {len(shape)}"))
        else:
            raise(ValueError(f"Shape must be int, list or np.array. Given: {type(shape)}"))
                
        self.mesh = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), np.arange(self.shape[2]))
        
        self.objects = []
        self.rotations = []
        self.built = False

    def add_ellipsoid(self, a: float, b: float, c: float, r: float, shift = [0,0,0]):
        self.objects.append({"a" : a, "b" : b, "c" : c, "r": r, "shift" : shift})
    
    def add_sphere(self, r: float, shift = [0,0,0]):
        self.add_ellipsoid(1, 1, 1, r, shift)
        
    def rotate(self, plane: str, angle: float, **kwargs: dict):
        """Rotates the entire mask in the given plane. Additional kwargs are passed to ndimage.rotate.

        Args:
            plane (str): Plane of rotation (xy, xz, yz)
            angle (float): Angle of rotation in degrees.
        """
        if self.built:
            raise RuntimeError("Can not rotate an object that has been built")
        match plane:
            case "xy":
                axes = (1, 0)
            case "xz":
                axes = (1, 2)
            case "yz":
                axes = (0, 2)
        rot_dict = {"axes" : axes, "angle": angle}
        rot_dict.update(kwargs)
        self.rotations.append(rot_dict)
        
    def build(self):
        mask = np.zeros(self.shape, dtype=np.int8)
        for obj in self.objects:
            obj_mask = make_ellipsoid(self.mesh, **obj)
            mask = np.logical_or(mask, obj_mask)
            
        if self.rotations:
            for rot in self.rotations:
                mask = np.where(ndi.rotate(mask.astype(np.float64), reshape = False, **rot) > .5, 1, 0).astype(np.int8)
            
        self.mask = mask
        self.volume = np.sum(mask)
        self.shell = np.array(mask, dtype = np.int8) - ndi.binary_erosion(np.array(mask, dtype = np.int8))
        self.area = np.sum(self.shell)
        self.built = True
        return mask

def make_background(mask_builder_obj, function, A_C, factor, norm = True):
    
    bkg_mask = np.ones(mask_builder_obj.shape) - mask_builder_obj.mask
    if function == None:
        function = lambda x,y,z: np.int8(1)
    bkg = function(*mask_builder_obj.mesh).astype(np.float64)
    
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

#%%
def Rv_t_s(n, FWHM):
    sigma = np.float32(FWHM/(2*np.sqrt(2*np.log(2))))
    return 1 - sigma/(np.array(n) * np.sqrt(2 * np.pi))

def sim(circ_r, FWHM, A_C = 1, background = 0.2, bkg_f = None, mask_size = 64, _gaussian_size = 32, plot = False):
        mask = mask_builder(mask_size, 2)
        # mask.add_sphere(circ_r)
        mask.add_ellipsoid(0.5, 0.7, 0.5, circ_r)
        mask.add_ellipsoid(0.5, 0.5, 1.5, circ_r, shift = [4,-3 , 2])
        mask.rotate("xz", random.random() * 90, order=0, mode = "nearest")
        mask.rotate("yz", random.random() * 90, order=0, mode = "nearest")
        mask.build()
        
        
        sim_space = np.ones(mask.shape, dtype=np.float64) * mask.mask * A_C 
        if background:
            sim_space += make_background(mask, bkg_f, A_C, background)
        # sim_space = mask.env * A_C
        # psf = gaussian_psf(_gaussian_size, FWHM)
        # sim_space = signal.convolve(sim_space, psf, mode="same")
        sigma = np.float32(FWHM/(2*np.sqrt(2*np.log(2))))
        sim_space =ndi.gaussian_filter(sim_space, sigma, mode="mirror", radius=16)
        A_est = np.sum(sim_space * mask.mask) / mask.volume

        
        if plot:
            fix, ax = plt.subplots(1,1)
            ax.imshow(mask.mask[mask_size // 2])
            print(ndi.center_of_mass(mask.mask))
            fig, axes = plt.subplots(1, 3, figsize = [9,3])
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace = 0.25, hspace = 0.25)
            axes[0].imshow(sim_space[mask_size // 2,:,:])
            axes[0].set_ylabel("x")
            axes[0].set_xlabel("z")
            axes[1].imshow(sim_space[:,mask_size // 2,:])
            axes[1].set_xlabel("z")
            axes[1].set_ylabel("y")
            axes[2].imshow(sim_space[:,:, mask_size // 2])
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("y")
            
            fig.show()

        sr = {"FWHM":FWHM, 
                "volume" : mask.volume, 
                "volume2area" : mask.volume/mask.area,
                "A_est": A_est,
                "A_true": A_C,
                "analytical" : Rv_t_s(mask.volume/mask.area, FWHM) * A_C}
        if background:
            bkg = calculate_background(sim_space, mask.mask, int(2*FWHM), 3)
            sr.update(bkg)
        return pd.DataFrame(sr, index = [1])
    
    
# C = sim(32, 4, A_C = 1, background = .2, bkg_f = lambda x, y, z: z, plot=True, _mask_size = 64)
# C = sim(16, 5, A_C = 2, background = 0.2, bkg_f = lambda x, y, z: np.where(x < 31, 2, 1), plot=True)
# C = sim(16, 2, A_C = 1, background = .5, bkg_f = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2), plot=True)

#%%
from tqdm import tqdm
import multiprocessing as mp
import time
def wrapper_sim(X):
    return sim(*X)

def run_simulation(arr_FWHM, arr_r, arr_A, mp_processes = os.cpu_count() // 2, mp_chunksize = 1):
    # Inputs must be zipped and passed through a wrapper to be unpacked
    ziped_input = zip(arr_r, arr_FWHM, arr_A)
    
    pool = mp.Pool(mp_processes)
    print(f"Simulating {len(arr_FWHM)} points")
    print(f"Launching {mp_processes} threads \n")
    start = time.time() 
    # Arrays are passed to a process pool through tqdm to make the process bar
    # Chunk size might effekt performance, no major difference noted so far
    # imap_unordered is lazy and returns generator, as compared to map, saving memory
    # starmap does not easily support progress bar

    results = pool.imap_unordered(wrapper_sim, tqdm(ziped_input, total = len(arr_r)), chunksize=mp_chunksize)
    pool.close()
    pool.join()
    
    
    
    stop = time.time()
    tot_time = stop - start
    print(f"\nAll threads completed! Total time: {int(tot_time // 3600)} h, {int(tot_time // 60)} min, {round(tot_time % 60)} s")
    data = pd.concat(results, ignore_index=True)
    data.index = (np.arange(len(data)))
    # print("\n",data.describe())
    # print(data.head())
    return data
    

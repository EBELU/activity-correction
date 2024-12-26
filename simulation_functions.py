#!/home/eewa/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:29:14 2024

@author: Erik Ewald
"""

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.signal as signal
import sys
import numba as nb
from numpy.random import rand
import pandas as pd

def plot_voxels(obj):
    fig_3d = plt.figure()
    ax3d = fig_3d.add_subplot(projection='3d')

    ax3d.voxels(obj)

@nb.njit
def make_ellipsoid(mesh_L, a, b, c, r, shift = [0,0,0]):
    xx, yy, zz = mesh_L
    px, py, pz = len(xx)/2, len(yy)/2, len(zz)/2
    xx, yy, zz = xx - px - shift[0], yy - py - shift[1], zz - pz - shift[2]
    return np.where((xx / a)**2 + (yy / b)**2 + (zz / c)**2 <= r**2, np.int8(1), np.int8(0))


def gaussian_psf(n, FWHM, norm=True):
    if np.isscalar(n):
        n = n * np.ones(3)
    if np.isscalar(FWHM):
        sigma = np.float32(FWHM/(2*np.sqrt(2*np.log(2))))
        FWHM = FWHM * np.ones(3)
        sigma = sigma * np.ones(3)

    x = np.arange(n[0], dtype=np.float64)
    x -= np.mean(x)

    y = np.arange(n[1], dtype=np.float64)
    y -= np.mean(y)
    
    
    z = np.arange(n[2], dtype=np.float64)
    z -= np.mean(z)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    psf = np.exp(-(xx**2/(2*sigma[0]**2) + yy**2/(2*sigma[1]**2) + zz**2/(2*sigma[2]**2)))
    
    if norm:
        psf /= np.sum(psf)
    
    return psf


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

    def add_ellipsoid(self, a, b, c, r, shift = [0,0,0]):
        self.objects.append({"a" : a, "b" : b, "c" : c, "r": r, "shift" : shift})
    
    def add_sphere(self, r, shift = [0,0,0]):
        self.add_ellipsoid(1, 1, 1, r, shift)
        
    def rotate(self, plane, angle, **kwargs):
        match plane:
            case "xy":
                axes = (0, 1)
            case "xz":
                axes = (0, 2)
            case "yz":
                axes = (1, 2)
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
    
    y_shell = np.zeros_like(shell_mask)
    y_shell[int(com[1]):,:,:] = shell_mask[int(com[1]):,:,:]
    
    z_shell = np.zeros_like(shell_mask)
    z_shell[:,:, int(com[2]):] = shell_mask[:,:, int(com[2]):] 
    
    
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
    
    return {"b_x":np.sum(image*x_shell) / np.sum(x_shell),
            "b_nx":np.sum(image*(shell_mask - x_shell)) / np.sum((shell_mask - x_shell)),
            "b_y":np.sum(image*y_shell) / np.sum(y_shell),
            "b_ny":np.sum(image*(shell_mask - y_shell)) / np.sum((shell_mask - y_shell)),
            "b_z:":np.sum(image*z_shell) / np.sum(z_shell),
            "b_nz":np.sum(image*(shell_mask - z_shell)) / np.sum((shell_mask - z_shell))
        }

#%%
def Rv_t_s(n, FWHM):
    sigma = np.float32(FWHM/(2*np.sqrt(2*np.log(2))))
    return 1 - sigma/(np.array(n) * np.sqrt(2 * np.pi))

def sim(circ_r, FWHM, A_C = 1, background = 0.2, bkg_f = None, mask_size = 64, _gaussian_size = 32, plot = False):
        mask = mask_builder(mask_size, 2)
        # mask.add_sphere(circ_r)
        mask.add_ellipsoid(0.5, 0.7, 0.5, circ_r)
        # mask.add_ellipsoid(0.5, 0.5, 1.5, circ_r, shift = [4,-3 , 2])
        # mask.rotate("xz", 45, order=2, mode = "nearest")
        # mask.rotate("yz", 30, order=2, mode = "nearest")
        mask.build()
        
        
        sim_space = np.ones(mask.shape, dtype=np.float64) * mask.mask * A_C 
        if background:
            sim_space += make_background(mask, bkg_f, A_C, background)
        # sim_space = mask.env * A_C
        psf = gaussian_psf(_gaussian_size, FWHM)
        sim_space = signal.convolve(sim_space, psf, mode="same")
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
# C = sim(16, 3, A_C = 2, background = 0.2, bkg_f = lambda x, y, z: np.where(x < 31, 2, 1), plot=False)
# C = sim(16, 2, A_C = 1, background = .5, bkg_f = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2), plot=True)

#%%
from tqdm import tqdm
import multiprocessing as mp
import time
def wrapper_sim(X):
    return sim(*X)

def run_simulation(arr_FWHM, arr_r, arr_A, mp_processes = os.cpu_count(), mp_chunksize = 1):
    # Inputs must be zipped and passed through a wrapper to be unpacked
    ziped_input = zip(arr_r, arr_FWHM, arr_A)
    
    pool = mp.Pool(mp_processes)
    print(f"Simulating {len(arr_FWHM)} points")
    print(f"Launching {mp_processes} threads \n")
    start = time.time() 
    # Arrays are passed to a process pool thorugh tqdm to make the process bar
    # Chunk size might effekt performence, no major difference noted so far
    # imap_unordered is lazy and returns generotor, as compared to map, saving memory
    # starmap does not easily support progress bar

    results = pool.imap_unordered(wrapper_sim, tqdm(ziped_input, total = len(arr_r)), chunksize=mp_chunksize)
    pool.close()
    pool.join()
    
    
    
    stop = time.time()
    tot_time = stop - start
    print(f"\nAll threads completed! Total time: {tot_time // 3600} h, {tot_time // 60} min, {round(tot_time % 60)} s")
    data = pd.concat(results, ignore_index=True)
    data.index = (np.arange(len(data)))
    # print("\n",data.describe())
    # print(data.head())
    return data
    
if __name__ == "__main__":
    cpu_count = os.cpu_count() - 1
    elements = 1000
    
    rand_FWHM = (rand(elements) * 3) + 3
    rand_radius = (rand(elements) * 6) + 7
    rand_A = (rand(elements) * 40)
    
    data = run_simulation(rand_FWHM, rand_radius, rand_A)

    #%%


                     
def compare_data(data):
    err_fig, axes = plt.subplots(3, 1)
    err_fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace = 0.15, hspace = 0.25)
    axes = axes.flatten()
    axes[0].plot(data["volume2area"], data["A_est"], label = "est", marker = "o", linestyle = "none")
    axes[0].plot(data["volume2area"], data["analytical"], label = "analytical", marker = "o", linestyle = "none")
    axes[0].legend()
    axes[1].plot(data["volume2area"], abs(data["A_est"] - data["analytical"])/data["A_est"] * 100, label = "est", marker = "o", linestyle = "none")
    # plt.plot(data["volume2area"], data["analytical"], label = "analytical", marker = "o", linestyle = "none")
    axes[1].legend()
    axes[2].plot(data["FWHM"], abs(data["A_est"] - data["analytical"])/data["A_est"] * 100, label = "est", marker = "o", linestyle = "none")  
    axes[0].set_ylabel("Estimated activity")
    axes[1].set_ylabel("Error relative A_est [%]")
    axes[2].set_ylabel("Error relative A-est [%]")
    axes[0].set_xlabel("Volume to surface")
    axes[1].set_xlabel("Volume to surface")
    axes[2].set_xlabel("FWHM")
    # ax = plt.figure().add_subplot(projection='3d')
    
    # ax.voxels(shell)
    # ax.voxels(elp, alpha = 0.2, color = "red")
    
    # try:
    #     plt.plot()
    # except:
    #     pass

#%%
if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    df_train = data.dropna().sample(frac=0.8, random_state=0)
    df_test = data.dropna().drop(df_train.index)
    
    train_labels = df_train.pop("A_true")
    test_labels = df_test.pop("A_true")
    
    normalizer = tf.keras.layers.Normalization(axis=-1)
    
    normalizer.adapt(np.array(df_train))
    print(normalizer.mean.numpy())
    
    def build_and_compile_model(norm):
      model = keras.Sequential([
          norm,
          layers.Dense(32, activation='relu'),
          layers.Dense(64, activation='relu'),
          layers.Dense(1)
      ])
    
      model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.005))
      return model
    
    
    dnn_model = build_and_compile_model(normalizer)

#%%
if __name__ == "__main__":
    history = dnn_model.fit(
        df_train,
        train_labels,
        validation_split=0.2,
        verbose=1, epochs=200)

#%%
if __name__ == "__main__":
    def plot_loss(history):
      plt.plot(history.history['loss'], label='loss')
      plt.plot(history.history['val_loss'], label='validation_loss')
      plt.ylim([0, 15000])
      plt.xlabel('Epoch')
      plt.ylabel('Error [MPG]')
      # plt.yscale()
      plt.legend()
      plt.grid(True)
      
    plot_loss(history)

#%%
if __name__ == "__main__":
    dnn_model.evaluate(df_test, test_labels, verbose=1)
    
    x = tf.linspace(1, 20, 420)
    y = dnn_model.predict(df_test)
    test_result = dnn_model.evaluate(df_test, test_labels)
    print(test_result)
    
    plt.plot(list(test_labels.index), y, label = "Model", linewidth = 2)
    plt.plot(list(test_labels.index), test_labels, label = "Test Labels")
    plt.legend()
    plt.ylabel("True activity")

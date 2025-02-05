#!/home/eewa/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:16:30 2025

@author: Erik Ewald
"""

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.ndimage as ndi
import multiprocessing as mp
import pandas as pd
import datetime
from tqdm import tqdm

from importlib.util import find_spec
_use_itk = True

if find_spec("SimpleITK") and _use_itk:
    from mask_builder_sitk import mask_builder
else:
    from mask_builder_sp import mask_builder

from misc import Rv_t_s
from background_functions import bkg_functions, make_background, calculate_background


def worker(arg_dict):
    """
    NOT INTENDED TO BE USED OUTSIDE THIS FILE
    Performs the smulation and serves to distribute the work across several processes.
    """
    
    # Use the given target constructor
    # Mask base is already decleared and now is only copied
    # "target_args" are unpacked as positionals
    mask = arg_dict["target_constructor"](mask_builder(*arg_dict["target_base"]), *arg_dict["target_args"])
    
    # Check if the target mask was built in the constructor
    if not mask.built:
        mask.build()
    
    # Generate the target object in space and set the activity concentration
    sim_space = np.ones(mask.shape, dtype=np.float64) * mask.mask * arg_dict["A_C"]
    
    sim_space += make_background(mask, lambda x,y,z: x, arg_dict["A_C"], arg_dict["bkg_A"])
    
    # Apply Gaussian filter
    sigma = np.float32(arg_dict["FWHM"]/(2*np.sqrt(2*np.log(2))))
    sim_space = ndi.gaussian_filter(sim_space, sigma, mode="mirror", radius=16)
    
    # Apply the mask to the blured image and sum the activity inside the mask
    A_est = np.sum(sim_space * mask.mask) / mask.volume
    
    # Merge all data of interest into a dict for easy conversion to pandas DataFrame
    sr = {"FWHM":arg_dict["FWHM"], 
            "volume" : mask.volume, 
            "volume2area" : mask.volume/mask.area,
            "A_est": A_est,
            "A_true": arg_dict["A_C"],
            "analytical" : Rv_t_s(mask.volume/mask.area, arg_dict["FWHM"]) * arg_dict["A_C"]}
    
    bkg = calculate_background(sim_space, mask.mask, int(2*arg_dict["FWHM"]), 3)
    sr.update(bkg)
    
    # plt.imshow((mask.mask * sim_space)[:,:,32])
    return pd.DataFrame(sr, index = [0])
    
def mask_constructor(mask: mask_builder, r):
    mask.add_sphere(r* 0.3, shift = [-2, 2, -3])
    mask.add_ellipsoid(0.5, 0.7, 0.5, r)
    mask.add_ellipsoid(0.5, 0.5, 1.5, r, shift = [4,-3 , 2])
    mask.add_ellipsoid(0.4, 1.2, 0.6, r, shift = [2,-1 , 0])
    mask.rotate(45, 30, 45)
    return mask

class simulation:
    def __init__(self, size, spacing, expectedElements):
        """Instantiate a simulation.

        Args:
            size (int): Number of voxels in each dimension.
            spacing (float): Side length of the cubic voxels in physical space.
            expectedElements (int): Number of elements to run. Used to check the length of provided arrays.
        """        
        self.size = size
        self.spacing = spacing
        self.expected_elements = expectedElements
        self.xx, self.yy, self.zz = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
        
        self.tests = {"has_target_constructor" : False,
                      "has_target_arguments" : False,
                      "has_activity_concentration" : False,
                      "has_FWHM" : False}
        
        self.run_complete = False
        self.absoulte_background = False
        
        self._bkg_activity = np.zeros(expectedElements) + .2
    def SetTargetConstructor(self, func):
        """Takes a function as an argument that will be called for each iteration of the simulation to construct the target. 

        Args:
            func : A function that must take a mask_builder object as a fist positional argument and other arguments as positional. See SetTargetArgs for providing arguments.
        """                
        self._target_constructor = func
        self.tests["has_target_constructor"] = True
        
        
    def SetTargetArgs(self, *args):
        """ Takes an arbitrary number of arrays that provide positional arguments to the function set with SetTargetConstructor. The arguments are supplies one from each array per iteration in the positional order they are given to this function.
        """        
        args = list(args)
        for i, arg in enumerate(args):
            if np.isscalar(arg):
                args[i] = np.ones(self.expected_elements) * arg
            else:
                self._check_array_length(arg)
        self._target_constructor_args = args
        self.tests["has_target_arguments"] = True
        
    
    def SetBackgroundFunction(self, func = "random", randomList = None):
        """ Sets the function function to generate the background in the simulation. The function can be explicitly be given of picked from the predefined function. 
            
        Args:
            func (function): A function that will be passed mesh-grids for the x-,y- and z-directions. If set to random function will be picked from predefined functions. A list of integers must be provided to index the list of predefined functions.
            randomList (list): An iterable containing ints to get background function from bkg_functions.get_from_idx
        """
        if func == "random":
            pass
        else:
            self._bkg_function = func
        self.tests["has_background_fn"] = True
        self.tests["has_background_activity"] = False
            
    def SetBackgroundActivity(self, bkgActivity, absolute = False):
        if np.isscalar(bkgActivity):
            bkgActivity = np.ones(self.expected_elements) * bkgActivity
        else:
            self._check_array_length(bkgActivity)
        
        self._bkg_activity = bkgActivity
        self.tests["has_background_activity"] = True
        
        if absolute:
            self.absoulte_background = True
            
    def SetActivityConcentration(self, A_C):
        """Set activity concentration of target for each iteration of the simulation.

        Args:
            A_C (Array or scalar): Array with activity concentration for each iteration of the simulation or scalar to set constant activity concentration.
        """        
        if np.isscalar(A_C):
            A_C = np.ones(self.expected_elements) * A_C
        else:
            self._check_array_length(A_C)
            
        self.A_C = A_C
        self.tests["has_activity_concentration"] = True
        
        
    def SetFWHM(self, FWHM):
        """Set the FWHM for the Gaussian filter applied in the simulation.

        Args:
            FWHM (array or scalar): Array with FWHM for each iteration of the simulation or scalar to set constant FWHM.
        """        
        if np.isscalar(FWHM):
            FWHM = np.ones(self.expected_elements) * FWHM
        else:
            self._check_array_length(FWHM)
        self.FWHM = FWHM
        self.tests["has_FWHM"] = True
            
        
    def SaveParameters(self, fileName = None):
        """
        Unfinished
        """
        dictonary = {}
        try:
            for i, arg in enumerate(self._target_constructor_args):
                dictonary[f"constructor_arg_{i}"] = arg
        except(AttributeError):
            print("No constructor args, Skipping")
        
        try:
            dictonary["FWHM"] = self.FWHM
        except(AttributeError):
            print("No FWHM, skipping")
            
        try:
            dictonary["A_C"] = self.A_C
        except(AttributeError):
            print("No activity concentration, skipping")  
        
        print(pd.DataFrame(dictonary))
        pass
    
    def SaveResults(self, fileName = None):
        if not fileName:
            fileName = "simulation_results_" + str(datetime.datetime.now())
        
        if self.run_complete:
            self.results.to_csv(fileName)
        else:
            raise RuntimeError("Can not save simulated parameters as no simulation has been completed")
        print("Results saved to:", fileName)
        
    def Run(self, multiprocessing = True, mpProcesses = os.cpu_count() // 2):
        # Check tests before prociding with simulation
        for test, state in self.tests.items():
            break
            if not state:
                raise RuntimeError(f"Test '{test}' failed! Check that corresponding value has been set correctly.")
                
        if not self.absoulte_background:
            # If the background is not absolute it is a fraction of the activity concentration
            # Update the background with the given activity concentration
            self._bkg_activity *= self.A_C
            
        # Bundle arguments that are to be passed to the worker into a dicts in a list
        worker_dicts = [{"target_args" : target_args,
                         "A_C": self.A_C[i],
                         "FWHM": self.FWHM[i],
                         "bkg_A": self._bkg_activity[i],
                         "target_constructor": self._target_constructor,
                         "target_base": (self.size, self.spacing)}
                        for i, target_args in enumerate(np.column_stack(self._target_constructor_args))]

    
        if multiprocessing:
            pool = mp.Pool(mpProcesses)
            
            results = pool.imap(worker, tqdm(worker_dicts, total = self.expected_elements))
            
            pool.close()
            pool.join()
        else:
            results = [worker(dictionary) for dictionary in tqdm(worker_dicts, total = self.expected_elements)]



        df = pd.concat(results, ignore_index=True)
        self.run_complete = True
        self.results = df
        return df
            
            
            
    def _check_array_length(self, arr):
        if len(arr) != self.expected_elements:
            raise AttributeError(f"Argument list must have the expected length! ({self.expected_elements}) Has {len(arr)}")
            
        
            
if __name__ == "__main__":   
    elements = 1500
    S = simulation(64, 1, elements)
    
    import numpy.random as nprand
    
    nprand.seed(0)
    rand_FWHM = (nprand.rand(elements) * 3) + 3
    rand_radius = (nprand.rand(elements) * 6) + 7
    rand_A = (nprand.rand(elements) * 40)
    
    S.SetFWHM(rand_FWHM)
    S.SetActivityConcentration(rand_A)
    S.SetTargetArgs(rand_radius)
    S.SetTargetConstructor(mask_constructor)
    
    
    
    S.SaveParameters()
    
    res=S.Run(multiprocessing=True)
#%%


# def f(arg):
#     i = arg["mask_args"]
    
#     print(i, arg)
            
# A = np.arange(5)
# B = np.arange(5)
# C = ["A", "B", "C", "D", "E"]

# D = np.arange(5) + 5

# L = [{"mask_args" : mask_args, "A_C" : D[i], "bkg_fn" : C[i]} for i, mask_args in enumerate(zip(A,B))]


# pool = mp.Pool(2)

# Arrays are passed to a process pool through tqdm to make the process bar
# Chunk size might effekt performance, no major difference noted so far
# imap_unordered is lazy and returns generator, as compared to map, saving memory
# starmap does not easily support progress bar

# results = pool.map(f, L)


            

#%%

if __name__ == "__main__":
    import cProfile
    cProfile.run("S.Run(False)",  "profileing", "cumtime")
    
    import pstats
    p = pstats.Stats("profileing")
    print(p.sort_stats('cumulative').print_stats(15))


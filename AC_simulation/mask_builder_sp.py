# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:06:57 2025

@author: Erik Ewald
"""
import numpy as np
import scipy.ndimage as ndi


from importlib.util import find_spec
if find_spec("numba"):
    from make_ellipse_numba import make_ellipsoid
else:
    from make_ellipse_base import make_ellipsoid

class mask_builder:
    def __init__(self, size, spacing):
        self.size = size
        self.shape = np.ones(3, dtype=np.int8) * size
        self.xx, self.yy, self.zz = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
        
        self.objects = []
        self.rotations = []
        self.built = False

    def add_ellipsoid(self, a: float, b: float, c: float, r: float = 1, shift = [0,0,0]):
        self.objects.append({"a" : a, "b" : b, "c" : c, "r": r, "shift" : shift})
    
    def add_sphere(self, r: float, shift = [0,0,0]):
        self.add_ellipsoid(1, 1, 1, r, shift)
        
    def rotate(self, x = 0, y = 0, z = 0):
        if x:
            self.rotations.append({"axes" : (0, 2), "angle" : x})   
        if y:   
            self.rotations.append({"axes" : (1, 2), "angle" : y})  
        if z:
            self.rotations.append({"axes" : (0, 1), "angle" : z})  

        
    def build(self):
        mask = np.zeros(self.size * np.ones(3, dtype=np.int8), dtype=np.int8)
        for obj in self.objects:
            obj_mask = make_ellipsoid([self.xx, self.yy, self.zz], **obj)
            mask = np.logical_or(mask, obj_mask)
            
        if self.rotations:
            for rot in self.rotations:
                mask = ndi.rotate(mask.astype(np.float64), reshape = False, order = 0, mode = "nearest", **rot)
            
        self.mask = mask
        self.volume = np.sum(mask)
        self.shell = mask * ndi.binary_erosion(mask)
        self.area = np.sum(self.shell)
        self.built = True
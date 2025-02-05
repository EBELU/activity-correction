# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:06:59 2025

@author: Erik Ewald
"""

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from importlib.util import find_spec
if find_spec("numba"):
    from make_ellipse_numba import make_ellipsoid
else:
    from make_ellipse_base import make_ellipsoid
    

def sitk_rotation(image, rot_vec):
    # Create rotation matrices for each axis
    transform = sitk.Euler3DTransform()
    transform.SetRotation(*np.deg2rad(rot_vec))
    transform.SetCenter(np.array(image.GetSize()) / 2)
    
    
    # Get the image size and spacing
    size = image.GetSize()
    spacing = image.GetSpacing()
    
    # Perform the resampling using the combined rotation transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetSize(size)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    
    # Apply the transformation
    return resampler.Execute(image)

class mask_builder:
    def __init__(self, size, spacing):
        self.size, self.spacing = size, spacing
        self.xx, self.yy, self.zz = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
        self.shape = (size, size, size)
        
        self.objects = []
        self.rotation_vect = None
        self.built = False
        
    def add_ellipsoid(self, a: float, b: float, c: float, r: float, shift = [0,0,0]):
        self.objects.append({"a" : a, "b" : b, "c" : c, "r": r, "shift" : shift})
    
    def add_sphere(self, r: float, shift = [0,0,0]):
        self.add_ellipsoid(1, 1, 1, r, shift)
        
    def rotate(self, x, y, z):
        self.rotation_vect = [z, y, x]
        
    def build(self):
        mask = np.zeros(self.xx.shape, dtype=np.int8)
        for obj in self.objects:
            obj_mask = make_ellipsoid([self.xx, self.yy, self.zz], **obj)
            mask = np.logical_or(mask, obj_mask)
            
        image = sitk.GetImageFromArray(mask.astype(np.int16))
        
        if self.rotation_vect:
            image = sitk_rotation(image, self.rotation_vect)
        self.image = image
        self.mask = sitk.GetArrayViewFromImage(image)
        self.contour = sitk.GetArrayFromImage(sitk.BinaryContour(image, fullyConnected=True))
        self.volume = np.sum(self.mask.astype(np.float64))
        self.area = np.sum(self.contour.astype(np.float64))
        
        self.built = True
            
def f1():       
    global M1
    M1 = mask_builder(64, 1)
    
    M1.add_ellipsoid(1,1,2, 8)
    M1.add_ellipsoid(1,2,1, 6)       
    
    M1.rotate(*[0,0,45])    
    
    M1.build()     

    
    
            
if __name__ == "__main__":
    import cProfile
    cProfile.run("f1()",  "profileing", "cumtime")
    
    import pstats
    p = pstats.Stats("profileing")
    print(p.sort_stats('cumulative').print_stats(15))
    
    f1()
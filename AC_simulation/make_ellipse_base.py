# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:09:12 2025

@author: Erik Ewald
"""

import numpy as np


def make_ellipsoid(mesh_L, a, b, c, r, shift = [0,0,0]):
    xx, yy, zz = mesh_L
    px, py, pz = len(xx)/2, len(yy)/2, len(zz)/2
    xx, yy, zz = xx - px - shift[0], yy - py - shift[1], zz - pz - shift[2]
    return np.where((xx / a)**2 + (yy / b)**2 + (zz / c)**2 <= r**2, np.int8(1), np.int8(0))
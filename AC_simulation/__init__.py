# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:17:27 2025

@author: Erik Ewald
"""



from .simulation import simulation
from .background_functions import bkg_functions


from importlib.util import find_spec

if find_spec("SimpleITK"):
    from .mask_builder_sitk import mask_builder
else:
    from .mask_builder_sp import mask_builder



"""
Create Custom Colormaps

Reference for creating colorbars: 
https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import matplotlib as mpl
from matplotlib.colors import ListedColormap
import numpy as np
import copy


#---------------------------------------------------------------------------------------------------
# Generate Custom Colormaps
#---------------------------------------------------------------------------------------------------

def generate_cust_cmaps_dict():

    cmaps = {}

    # Add built-in colormaps
    for k in mpl.colormaps.keys():
        cmaps[k] = copy.deepcopy(mpl.colormaps.get_cmap(k))

    # RH colorbar designed for partial cloudiness
    reds = mpl.colormaps['Reds_r'].resampled(25)
    grays = mpl.colormaps['Greys'].resampled(10)
    RH_colors = np.zeros([24, 4])
    RH_colors[1:19, :] = reds(np.linspace(0, 1, 25)[:18])
    RH_colors[19:24, :] = grays(np.linspace(0, 1, 10)[:5])
    RH_colors[0, :] = [235/256, 12/256, 209/256, 1]  # Pink
    RH_colors[-1, :] = grays(np.linspace(0, 1, 10)[6])  # Dark grey
    RH_cmap = ListedColormap(RH_colors)
    cmaps['RH_cmap'] = RH_cmap

    return cmaps
    

"""
End colormaps.py
"""

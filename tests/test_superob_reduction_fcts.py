"""
Tests for superob_reduction_fcts.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pytest
import xarray as xr

#import pyDA_utils.superob_reduction_fcts as srf
import pyDA_utils.bufr as bufr
import pyDA_utils.map_proj as mp


#---------------------------------------------------------------------------------------------------
# Test Functions
#---------------------------------------------------------------------------------------------------

"""
General parameters for testing... will create a function later
"""

uas_fname = '/work2/noaa/wrfruc/murdzek/nature_run_spring/obs/uas_hspace_35km_ctrl/bogus_uas_csv/202204291300.rap.prepbufr.csv'
uas_csv = bufr.bufrCSV(uas_fname)

grid_fname = '/work2/noaa/wrfruc/murdzek/src/osse_ob_creator/fix_data/RRFS_grid_max.nc'
grid_ds = xr.open_dataset(grid_fname)

# Compare map projection function to RRFS (lat, lon) coordinates
lat1d = grid_ds['lat'].values.ravel()
lon1d = grid_ds['lon'].values.ravel() - 360
x1d, y1d = mp.ll_to_xy_lc(lat1d, lon1d, dx=3, knowni=899, knownj=529)
x2d_mp = np.reshape(x1d, grid_ds['lat'].shape)
y2d_mp = np.reshape(y1d, grid_ds['lat'].shape)
x2d_true, y2d_true = np.meshgrid(np.arange(1799), np.arange(1059))



"""
End test_superob_reduction_fcts.py
"""

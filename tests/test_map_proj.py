"""
Tests for map_proj.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr
import pytest

import pyDA_utils.map_proj as mp


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

def test_ll_to_xy_lc():
  
    # Load test data
    fname = './data/wrf_grid_3km.nc'
    wrf_ds = xr.open_dataset(fname)
 
    # Create 1D arrays
    x2d, y2d = np.meshgrid(np.arange(1799), np.arange(1059))
    lat1d = np.ravel(wrf_ds['XLAT'])
    lon1d = np.ravel(wrf_ds['XLONG'])
    x1d_true = np.ravel(x2d)
    y1d_true = np.ravel(y2d)

    # Perform map projection
    x1d, y1d = mp.ll_to_xy_lc(lat1d, lon1d, ref_lat=38.5, ref_lon=-97.5, truelat1=38.5, 
                              truelat2=38.5, stand_lon=-97.5, dx=3., e_we=1799, e_sn=1059, 
                              knowni=899, knownj=529)

    assert np.allclose(x1d, x1d_true, atol=0.0015)
    assert np.allclose(y1d, y1d_true, atol=0.0015)


"""
End test_map_proj.py
"""

"""
UPP Post Processing 

Class that provides additional post-processing of UPP fields

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr
import numpy as np
import metpy.calc as mc
from metpy.units import units
import metpy.constants as const


#---------------------------------------------------------------------------------------------------
# UPP Class
#---------------------------------------------------------------------------------------------------

class upp():
    """
    Class that handles UPP GRIB output

    Parameters
    ----------
    fname : string
        UPP GRIB file name

    """

    def __init__(self, fname):
        self.fname = fname
        self.ds = xr.open_dataset(fname, engine='pynio')
    
    
def compute_ceil_agl(ds, no_ceil=2e4, fields={'CEIL_LEGACY':'HGT_P0_L215_GLC0', 
                                              'CEIL_EXP1':'CEIL_P0_L215_GLC0',
                                              'CEIL_EXP2':'CEIL_P0_L2_GLC0'}):
    """
    Compute cloud ceilings AGL from cloud ceilings ASL

    Parameters
    ----------
    ds : xr.Dataset
        Input UPP Dataset
    no_ceil : float, optional
        File value for "no ceiling"
    fields : dictionary, optional
        Cloud ceiling fields. Key: New name, value: old name

    Returns
    -------
    ds : xr.Dataset
        UPP Dataset with cloud ceilings AGL added

    """

    # Extract surface terrain height
    terrain = mc.geopotential_to_height(ds['HGT_P0_L1_GLC0'].values * units.m * const.g).magnitude

    # Compute ceilings AGL
    for new_name in fields.keys():
        ds[new_name] = ds[fields[new_name]]
        ds[new_name].values = (mc.geopotential_to_height(ds[fields[new_name]].values * units.m * const.g).magnitude -
                               terrain)
        ds[new_name].attrs['long_name'] = 'ceiling height'
        ds[new_name].attrs['units'] = 'm AGL'

        # Set "no ceiling" (NaN or 2e4) to proper fill value
        ds[new_name].values[np.isnan(ds[fields[new_name]])] = no_ceil
        ds[new_name].values[np.isclose(ds[fields[new_name]].values, 2e4)] = no_ceil

    return ds


def compute_wspd_wdir(ds, u_field='UGRD_P0_L100_GLC0', v_field='VGRD_P0_L100_GLC0', 
                      wspd_field='WSPD', wdir_field='WDIR'):
    """
    Compute wind speed and direction from u and v wind components

    Parameters
    ----------
    ds : xr.Dataset
        Input UPP Dataset
    u_field : string, optional
        name of u-component field
    v_field : string, optional
        name of v-component field
    wspd_field : string, optional
        name of output wind speed field
    wdir_field : string, optional
        name of output wind direction field

    Returns
    -------
    ds : xr.Dataset
        UPP Dataset with wind speed and direction

    """

    ds[wspd_field] = np.sqrt(ds[u_field]**2 + ds[v_field]**2)
    ds[wspd_field].attrs = ds[u_field].attrs
    ds[wspd_field].attrs['long_name'] = 'wind speed'

    ds[wdir_field] = 90. - np.rad2deg(np.arctan2(-ds[v_field], -ds[u_field]))
    ds[wdir_field].attrs = ds[u_field].attrs
    ds[wdir_field].attrs['long_name'] = 'wind direction'
    ds[wdir_field].attrs['units'] = 'deg'

    return ds


"""
End upp_postprocess.py
"""

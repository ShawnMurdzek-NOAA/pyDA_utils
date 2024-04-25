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
    

def convert_gpm_msl_to_m_agl(ds, field, terrain_field='HGT_P0_L1_GLC0'):
    """
    Convert a height field from gpm MSL to m AGL

    Parameters
    ----------
    ds : xr.Dataset
        Input UPP Dataset
    field : str
        Field to convert from gpm MSL to m AGL
    terrain_field : str, optional
        Field corresponding to the terrain field, by default 'HGT_P0_L1_GLC0'

    Returns
    -------
    hgt_agl : array
        Input field, but with heights AGL

    """

    terrain = mc.geopotential_to_height(ds[terrain_field].values * units.m * const.g).magnitude
    hgt_agl = mc.geopotential_to_height(ds[field].values * units.m * const.g).magnitude - terrain

    return hgt_agl


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

    # Compute ceilings AGL
    for new_name in fields.keys():
        ds[new_name] = ds[fields[new_name]]
        ds[new_name].values = convert_gpm_msl_to_m_agl(ds, fields[new_name])
        ds[new_name].attrs['long_name'] = 'ceiling height'
        ds[new_name].attrs['units'] = 'm AGL'

        # Set "no ceiling" (NaN or 2e4) to proper fill value
        ds[new_name].values[np.isnan(ds[fields[new_name]])] = no_ceil
        ds[new_name].values[np.isclose(ds[fields[new_name]].values, 2e4)] = no_ceil

    return ds


"""
End upp_postprocess.py
"""

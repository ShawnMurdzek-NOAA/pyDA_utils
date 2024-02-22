"""
Functions for Computing Superobs from BUFR CSV Files

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr

import pyDA_utils.bufr as bufr
import pyDA_utils.map_proj as mp


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def assign_superobs(bufr_obj, grid_ds, obtype=136, 
                    grid_field_names={'x':'lon', 'y':'lat', 'z':'HGT_AGL'},
                    map_proj=mp.ll_to_xy_lc,
                    map_proj_kw={dx=3, knowni=899, knownj=529},
                    check_proj=True):
    """
    Assign each observation of a certain type to a superob

    Parameters
    ----------
    bufr_obj : bufr.bufrCSV object
        Raw observations
    grid_ds : xr.Dataset object
        Model grid information
    obtype : integer, optional
        Observation type to perform superobbing on
    grid_field_names : dictionary, optional
        Names of the x, y, and z field names from grid_ds. It is assumed that the x and y field 
        are 2D (with dimensions y, x) and the z field is 1D
    map_proj : function, optional
        Map projection function
    map_proj_kw : dictionary, optional
        Additional keywords passed to the map projection function
    check_proj : Boolean, optional
        Option to check whether the map projection is appropriate

    Returns
    -------
    superob_id : array of integers
        Array with a length equal to the number of entries in bufr_obj. Contains the ID of the 
        superob that each raw observation is assigned to

    """

    # First test to ensure that the map projection is appropriate
    if check_proj:     
        tol = 0.4
        x_rmse, y_rmse = mp.rmse_map_proj(grid_ds[grid_field_names['y']],
                                          grid_ds[grid_field_names['x']],
                                          proj=map_proj,
                                          proj_kw=map_proj_kw)
        if (x_rmse > tol) or (y_rmse > tol):
            print(f'X RMSE = {x_rmse}')
            print(f'Y RMSE = {y_rmse}')
            raise ValueError('Large difference between grid coordinates and map projection coordinates')

    #



"""
End superob_reduction_fcts.py
"""

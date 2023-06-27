"""
Utilities for Creating Synthetic Observations

shawn.s.murdzek@noaa.gov
Date Created: 22 November 2022
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def determine_twgt(wrf_hr, dhr):
    """
    Compute the weight and index needed for time interpolation using 1-D linear interpolation

    Inputs
    ------
    wrf_hr : array
        Decimal hours for each WRF UPP file
    dhr : float
        Decimal hour for the given observation

    Returns
    -------
    ihr : integer
        Index for wrf_hr associated with twgt 
    twgt : float
        Interpolation weight associated with ihr

    """

    ihr = np.where((wrf_hr - dhr) <= 0)[0][-1]
    twgt = (wrf_hr[ihr+1] - dhr) / (wrf_hr[ihr+1] - wrf_hr[ihr])

    return ihr, twgt


def _linear_interp(val1, val2, wgt):
    """
    Interpolate linearly between two values given a specific weight

    Inputs
    ------
    val1 : float or array
        Value(s) with weight 'wgt'
    val2 : float or array
        Value(s) with weight '1-wgt'
    wgt : float
        Interpolation weight

    Returns
    -------
    val : float
        Interpolated value

    """

    return (val1 * wgt) + (val2 * (1. - wgt))


def _bilinear_interp_horiz(field, iwgt, jwgt, i0, j0, threeD=False, mask=np.ones(4)):
    """
    Interpolate the given field in the horizontal

    Inputs
    ------
    field : array 
        2D field to interpolate
    iwgt : float
        Interpolation weight for i0
    jwgt : float
        Interpolation weight fro j0
    i0 : integer
        Lower-left index for the first dimension of field
    j0 : integer
        Lower-left index for the second dimension of field
    threeD : boolean, optional
        Is this field actually 3D?
    mask : array, optional
        Omit certain gridpoints from interpolation (1 = keep, 0 = omit)
        Order: [(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)]

    Returns
    -------
    val : float
        Interpolated value

    """

    iind = np.array([i0,        i0,            i0+1,          i0+1])
    jind = np.array([j0,        j0+1,          j0,            j0+1])
    wgts = np.array([iwgt*jwgt, iwgt*(1-jwgt), (1-iwgt)*jwgt, (1-iwgt)*(1-jwgt)]) * mask

    if threeD:
        val = np.zeros(field.shape[0])
        for w, i, j in zip(wgts, iind, jind):
            val = val + w * field[:, i, j]
    else:
        val = 0.
        for w, i, j in zip(wgts, iind, jind):
            val = val + w * field[i, j]

    if mask.sum() < 4:
        val = val / np.sum(wgts)

    return val


def interp_x_y(field, ob_subset, threeD=False, mask=np.ones(4)):
    """
    Wrapper function for interpolation in x and y

    Inputs
    ------
    field : array
        Array containing WRF data to be interpolated
    ob_subset : series
        Pandas series containing a single observation
    threeD : boolean, optional
        Is this UPP field 3D?
    mask : array, optional
        Omit certain gridpoints from interpolation (1 = keep, 0 = omit)
        Order: [(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)]

    Returns
    -------
    val : float
        UPP output linearly interpolated in x and y to the observation location

    """

    iwgt = ob_subset['iwgt']
    jwgt = ob_subset['jwgt']
    i0 = ob_subset['i0']
    j0 = ob_subset['j0']

    val = _bilinear_interp_horiz(field, iwgt, jwgt, i0, j0, threeD=threeD, mask=mask)

    return val


def interp_x_y_t(wrf_data, wrf_hr, var, ob_subset, ihr, twgt, threeD=False, mask=np.ones(4)):
    """
    Wrapper function for interpolation in x, y, and t

    Inputs
    ------
    wrf_data : dictionary
        Dictionary of Xarray datasets containing UPP output data
    wrf_hr : list
        List of valid times (in decimal hours) for wrf_data. These are the keys for wrf_data
    var : string
        Variable from the UPP dataset to interpolate
    ob_subset : series
        Pandas series containing a single observation
    ihr : integer
        Index for the UPP dataset immediately before the observation time
    twgt : float
        Interpolation weight associated with the ihr UPP dataset
    threeD : boolean, optional
        Is this UPP field 3D?
    mask : array, optional
        Omit certain gridpoints from interpolation (1 = keep, 0 = omit)
        Order: [(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)]

    Returns
    -------
    val : float
        UPP output linearly interpolated in x, y, and t to the observation location

    """

    field1 = wrf_data[wrf_hr[ihr]][var]
    field2 = wrf_data[wrf_hr[ihr+1]][var]
    iwgt = ob_subset['iwgt']
    jwgt = ob_subset['jwgt']
    i0 = ob_subset['i0']
    j0 = ob_subset['j0']

    val = _linear_interp(_bilinear_interp_horiz(field1, iwgt, jwgt, i0, j0, threeD=threeD, mask=mask),
                         _bilinear_interp_horiz(field2, iwgt, jwgt, i0, j0, threeD=threeD, mask=mask), 
                         twgt)

    return val


def interp_x_y_z(wrf3d, ob_subset, wgt_name='kwgt', i0_name='ki0'):
    """
    Wrapper function for interpolation in x, y, and p or z

    Inputs
    ------
    wrf3d : array
        WRF 3D output array to interpolate
    ob_subset : series
        Pandas series containing a single observation
    wgt_name : string, optional
        Name of weight for vertical interpolation in ob_subset
    i0_name : string, optional
        Name of i0 index for vertical interpolation in ob_subset

    Returns
    -------
    val : float
        UPP output linearly interpolated in x, y, and p or z to the observation location

    """
    
    iwgt = ob_subset['iwgt']
    jwgt = ob_subset['jwgt']
    wgt = ob_subset[wgt_name]
    i0 = ob_subset['i0']
    j0 = ob_subset['j0']
    k0 = ob_subset[i0_name]

    val = _linear_interp(_bilinear_interp_horiz(wrf3d[k0, :, :], iwgt, jwgt, i0, j0),
                         _bilinear_interp_horiz(wrf3d[k0+1, :, :], iwgt, jwgt, i0, j0), wgt)

    return val


def interp_wrf_1d(a1d, ob_subset, i0_name='ki0', var='POB', itype='log'):
    """
    Wrapper function for interpolation in one dimension

    Inputs
    ------
    a1d : array
        1D array to interpolate
    ob_subset : series
        Pandas series containing a single observation
    i0_name : string, optional
        Column name in ob_subset corresponding to the i0 index
    var : string, optional
        Variable being interpolated in ob_subset
    itype : string, optional
        Interpolation type ('log' or 'linear')

    Returns
    -------
    val : float
        UPP output interpolated to the observation location
    wgt : float
        Weight used for interpolation

    """

    i0 = ob_subset[i0_name]
    if itype == 'log':
        wgt = np.log10(a1d[i0+1] / ob_subset[var]) / np.log10(a1d[i0+1] / a1d[i0])
    elif itype == 'linear':
        wgt = (a1d[i0+1] - ob_subset[var]) / (a1d[i0+1] - a1d[i0])
    val = _linear_interp(a1d[i0], a1d[i0+1], wgt)

    return val, wgt


"""
End create_ob_utils.py
"""

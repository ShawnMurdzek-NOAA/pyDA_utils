"""
Cloud Data Assimilation Forward Operator

Passed Arguments
----------------
    sys.argv[1] : BUFR CSV observation file name
    sys.argv[2] : UPP model output

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import sys
import xarray as xr
import copy
import datetime as dt
import numpy as np

from pyDA_utils import bufr


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def find_bufr_cloud_obs(bufr_obj, use_types=[180, 181, 182, 183, 184, 185, 186, 187, 188], anal_dhr=0.0):
    """
    Wrapper function for identifying BUFR cloud obs that can be used by the forward operator.

    Parameters
    ----------
    bufr_obj : bufr.BufrCSV
        Input BUFR CSV object
    use_types : list, optional
        BUFR types to use, by default [180, 181, 182, 183, 184, 185, 186, 187, 188]
    anal_dhr : float, optional
        Only retain obs from each TYP/SID combo closest to this valid time, by default 0.0

    Returns
    -------
    out_df : pd.DataFrame
        DataFrame containing cloud obs to be used by the forward operator

    """

    tmp_obj = copy.deepcopy(bufr_obj)

    # Only retain desired ob types
    tmp_obj.select_obtypes(use_types)

    # Only retain obs with cloud information
    tmp_obj.df = remove_missing_cld_ob(tmp_obj.df)

    # Only retain obs closest to analysis time
    tmp_obj.select_dhr(anal_dhr)

    out_df = tmp_obj.df
    out_df.reset_index(inplace=True, drop=True)

    return out_df 


def remove_missing_cld_ob(bufr_df):
    """
    Remove missing cloud obs from BUFR DataFrame

    Parameters
    ----------
    bufr_df : pd.DataFrame
        BUFR obs in DataFrame format

    Returns
    -------
    bufr_df : pd.DataFrame
        BUFR obs with missing cloud obs removed
        
    """

    bufr_df = bufr_df.loc[(~np.isnan(bufr_df['CLAM'])) & 
                          (~np.isclose(bufr_df['CLAM'], 10)) &
                          (~np.isclose(bufr_df['CLAM'], 14)) &
                          (~np.isclose(bufr_df['CLAM'], 15)) &
                          (~(~np.isclose(bufr_df['CLAM'], 0) & np.isnan(bufr_df['HOCB'])))]
    bufr_df.reset_index(inplace=True, drop=True)

    return bufr_df


if __name__ == '__main__':

    print('reading in BUFR obs...')
    bufr_obj = bufr.bufrCSV(sys.argv[1])

    print('reading in model UPP output...')
    model_ds = xr.open_dataset(sys.argv[2], engine='pynio')

    print('Identifying cloud obs...')
    cld_ob_df = find_bufr_cloud_obs(bufr_obj)


"""
End cloud_DA_forward_operator.py
"""
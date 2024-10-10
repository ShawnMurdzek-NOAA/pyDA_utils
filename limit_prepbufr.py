"""
Remove PrepBUFR Obs After a Certain Condition is Met

Useful for imposing realistic flight limits on UAS obs

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import copy

import pyDA_utils.bufr as bufr


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def remove_obs_after_lim(df, obtypes=[136, 236], nthres=3, field='cond', debug=0):
    """
    Remove all obs from a given TYP/SID combo after nthres values from field are True

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with output from a prepBUFR file
    obtypes : list of integers, optional
        Observation types to remove obs from
    nthres : integer, optional
        Number of obs in a row that have to have "field" == True before removing the rest of the obs
        from that particular TYP/SID
    field : string, optional
        Column of df containing the condition that must be met to remove obs
    debug : integer, optional
        Option to print extra output for debugging. Higher numbers means more output

    Returns
    -------
    df : pd.DataFrame
        DataFrame with obs removed

    """

    idx_drop = []
    for t in obtypes:
        all_sid = np.unique(df.loc[df['TYP'] == t, 'SID'].values)
        for s in all_sid:
            tmp_df = df.loc[np.logical_and(df['TYP'] == t, df['SID'] == s)].copy()
            tmp_df.sort_values('DHR', inplace=True)
            if debug > 0:
                print(f"{s} {t}")
                print(f"len(tmp_df) = {len(tmp_df)}")
                print(f"sum(tmp_df[field]) = {np.sum(tmp_df[field])}")
            roll_cond = tmp_df['cond'].rolling(nthres).sum()
            idx_cond = np.where(roll_cond == nthres)[0]
            if len(idx_cond) > 0:
                if debug > 0: print(f"adding indices for {t} {s}")
                idx_drop = idx_drop + list(tmp_df.index[idx_cond[0]:].values)

    # Drop indices
    if debug > 0: print(f"len(idx_drop) = {len(idx_drop)}")
    df.drop(idx_drop, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def wspd_limit(bufr_obj, lim=15, wind_type=236, match_type=136, match_kw={}):
    """
    Flag observations that exceed a wind speed threshold

    Parameters
    ----------
    bufr_obj : bufr.bufrCSV object
        BUFR CSV object
    lim : integer, optional
        Wind speed limit (m/s)
    wind_type : integer, optional
        Kinematic observation type
    match_type : integer, optional
        Thermodynamic observation type to copy the "cond" field to
    match_kw : dictionary, optional
        Additional keyword arguments passed to the match_types method

    Returns
    -------
    bufr_obj : bufr.bufrCSV object
        BUFR CSV object with an extra column indicating whether the wind speed threshold was met

    """

    # Compute wind speeds
    bufr_obj.df = bufr.compute_wspd_wdir(bufr_obj.df)

    # Flag wind speed limit
    bufr_obj.df['cond'] = (bufr_obj.df['WSPD'] > lim) * (bufr_obj.df['TYP'] == wind_type)

    # Match to another (thermodynamic) ob type
    bufr_obj.match_types(match_type, wind_type, copy_fields=['cond'], **match_kw)

    return bufr_obj


def detect_icing(bufr_obj, tob_lim=2, rh_lim=90, thermo_type=136, match_type=236, match_kw={}):
    """
    Detect icing conditions using a T and RH threshold

    Parameters
    ----------
    bufr_obj : bufr.bufrCSV object
        BUFR CSV object
    tob_lim : integer, optional
        Temperature limit (deg C)
    rh_lim : integer, optional
        Relative humidity limit (%)
    thermo_type : integer, optional
        Thermodynamic observation type
    match_type : integer, optional
        Kinematic observation type to copy the "cond" field to
    match_kw : dictionary, optional
        Additional keyword arguments passed to the match_types method

    Returns
    -------
    bufr_obj : bufr.bufrCSV object
        BUFR CSV object with an extra column indicating whether the icing conditions were met
    """

    # Compute RH
    bufr_obj.df = bufr.compute_RH(bufr_obj.df)

    # Flag rows that meet the icing criteria
    bufr_obj.df['cond'] = ((bufr_obj.df['TOB'] < tob_lim) * (bufr_obj.df['RHOB'] > rh_lim) *
                           (bufr_obj.df['TYP'] == thermo_type))

    # Match to another (kinematic) ob type
    bufr_obj.match_types(match_type, thermo_type, copy_fields=['cond'], **match_kw)

    return bufr_obj


"""
limit_prepbufr.py
"""

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
import datetime as dt

import pyDA_utils.bufr as bufr


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def remove_obs_after_lim(df, obtype, match_type=[136], match_fields=[], nthres=3, field='cond', debug=0):
    """
    Remove all obs from a given TYP/SID combo after nthres values from field are True

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with output from a prepBUFR file
    obtype : integer
        Observation type associated with the "field" column
    match_type : list of integers, optional
        Observation types to remove obs from in addition to obtype. Obs are removed starting with
        the first DHR from obtype where "field" == 1
    match_fields : list of strings, optional
        Fields used to match the observations from match_type to obtype. In addition to these fields,
        the TYP and SID fields must match.
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
    all_sid = np.unique(df.loc[df['TYP'] == obtype, 'SID'].values)
    for s in all_sid:
        tmp_df = df.loc[np.logical_and(df['TYP'] == obtype, df['SID'] == s)].copy()
        tmp_df.sort_values('DHR', inplace=True)
        if debug > 0:
            print(f"{s}")
            print(f"len(tmp_df) = {len(tmp_df)}")
            print(f"sum(tmp_df[field]) = {np.sum(tmp_df[field])}")
        roll_cond = tmp_df['cond'].rolling(nthres).sum()
        idx_cond = np.where(roll_cond == nthres)[0]
        if len(idx_cond) > 0:
            if debug > 0: print(f"adding indices for {obtype} {s}")
            idx_drop = idx_drop + list(tmp_df.index[idx_cond[0]:].values)

            # Match to other observation type
            for t in match_type:
                dhr = tmp_df.loc[tmp_df.index[idx_cond[0]], 'DHR']
                match_cond = ((df['TYP'] == t) * (df['SID'] == s) * (df['DHR'] >= dhr))
                for f in match_fields:
                    match_cond = match_cond * (df[f] == tmp_df.loc[tmp_df.index[idx_cond[0]], f])
                if debug > 0:
                    print(f"DHR to start dropping = {dhr}")
                    print(f"adding {np.sum(match_cond)} indices for {t} {s}")
                if np.sum(match_cond) > 0:
                    idx_drop = idx_drop + list(df.index[match_cond].values)

    # Drop indices
    if debug > 0: print(f"len(idx_drop) = {len(idx_drop)}")
    df.drop(idx_drop, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def wspd_limit(bufr_obj, lim=15, wind_type=236, verbose=0):
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

    Returns
    -------
    bufr_obj : bufr.bufrCSV object
        BUFR CSV object with an extra column indicating whether the wind speed threshold was met

    """

    if verbose > 0: print(f"[wspd_limit] start =", dt.datetime.now())

    # Compute wind speeds
    bufr_obj.df = bufr.compute_wspd_wdir(bufr_obj.df)
    if verbose > 0: print(f"[wspd_limit] done computing WSPD =", dt.datetime.now())

    # Flag wind speed limit
    bufr_obj.df['cond'] = (bufr_obj.df['WSPD'] > lim) * (bufr_obj.df['TYP'] == wind_type)
    if verbose > 0: print(f"[wspd_limit] done computing cond =", dt.datetime.now())

    return bufr_obj


def detect_icing(bufr_obj, tob_lim=2, rh_lim=90, thermo_type=136):
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

    return bufr_obj


"""
limit_prepbufr.py
"""

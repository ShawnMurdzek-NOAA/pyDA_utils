"""
Functions for Manipulating Various GSI Input and Output Files

shawn.s.murdzek@noaa.gov
Date Created: 18 May 2023
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr
import pandas as pd
import numpy as np
import scipy.stats as ss
import metpy.calc as mc
from metpy.units import units
import metpy.constants as const
import scipy.interpolate as si


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def read_errtable(fname):
    """
    Parse out observation errors from an errtable file in GSI

    Parameters
    ----------
    fname : string
        Name of errtable text file

    Returns
    -------
    errors : dictionary
        A dictionary of pd.DataFrame objects containing the observation errors

    Notes
    -----
    More information about the errtable format in GSI can be found here: 
    https://dtcenter.ucar.edu/com-GSI/users/docs/users_guide/html_v3.7/gsi_ch4.html#conventional-observation-errors

    """

    # Extract contents of file
    fptr = open(fname, 'r')
    contents = fptr.readlines()
    fptr.close()

    # Loop over each line
    errors = {}
    headers = ['prs', 'Terr', 'RHerr', 'UVerr', 'PSerr', 'PWerr']
    for l in contents:
        if l[5:21] == 'OBSERVATION TYPE':
            key = int(l[1:4])
            errors[key] = {}
            for h in headers:
                errors[key][h] = []
        else:
            vals = l.strip().split(' ')
            for k, h in enumerate(headers):
                errors[key][h].append(float(vals[k]))

    # Convert to DataFrame
    for key in errors.keys():
        errors[key] = pd.DataFrame(errors[key])
        for h in headers[1:]:
            errors[key][h].where(errors[key][h] < 5e8, inplace=True)

    return errors


def write_errtable(fname, errors):
    """
    Write observation error variance to a GSI errtable file

    Parameters
    ----------
    fname : string
        Name of errtable text file
    errors : dictionary
        A dictionary of pd.DataFrame objects containing the observation errors. First key is ob ID
        and second key is variable (e.g., Terr, RHerr, etc.)

    Returns
    -------
    None

    Notes
    -----
    More information about the errtable format in GSI can be found here: 
    https://dtcenter.ucar.edu/com-GSI/users/docs/users_guide/html_v3.7/gsi_ch4.html#conventional-observation-errors

    """

    # Open file
    fptr = open(fname, 'w')

    # Write to file
    headers = ['prs', 'Terr', 'RHerr', 'UVerr', 'PSerr', 'PWerr']
    ob_types = list(errors.keys())
    for o in ob_types:
        fptr.write(' %d OBSERVATION TYPE\n' % o)
        for h in headers:
            errors[o][h][np.isnan(errors[o][h])] = 0.1e10
        for j in range(len(errors[o]['prs'])):
            line = ' '
            for h in headers: 
                tmp = '%.5e' % (errors[o][h][j]*10)
                line = line + ' 0.%s%sE%s' % (tmp[0], tmp[2:6], tmp[8:])
            fptr.write('%s\n' % line)

    fptr.close()

    return None


def read_sfcobs_uselist(fname):
    """
    Read GSD surface obs uselist file as a DataFrame

    Parameters
    ----------
    fname : string
        GSD surface obs uselist filename

    Returns
    -------
    uselist_df : pd.DataFrame
        Contents of the uselist as a DataFrame

    """

    columns = ['SID', 'W_flag', 'T_flag', 'Td_flag', 'PRVSTG', 
               'T_N', 'T_avg', 'T_bias', 'T_std', 
               'S_N', 'S_avg', 'S_bias', 'S_std', 
               'W_N', 'W_avg', 'W_bias', 'W_std', 
               'Td_N', 'Td_avg', 'Td_bias', 'Td_std', 
               'start_date', 'start_time', 'end_date', 'end_time']

    uselist_df = pd.read_csv(fname, comment=';', names=columns, delim_whitespace=True)

    return uselist_df


def read_diag(fnames, mesonet_uselist=None, ftype='netcdf', date_time=[None]):
    """
    Read a series of GSI diag files, concatenate, and save into a DataFrame

    Parameters
    ----------
    fnames : list of strings
        GSI diag file names
    mesonet_uselist : string, optional
        GSD sfcobs uselist. Used to add a new column specifying the network for each mesonet site
    ftype : string, optional
        GSI diag file type ('netcdf' or 'text') 
    date_time : list of integers
        Date and time of the analysis reported in the diag file (YYYYMMDDHH). Only used if ftype = 
        'text'.

    Returns
    -------
    diag_out : pd.DataFrame
        GSI diag output in DataFrame format

    """

    # Read in each diag file and convert to DataFrame
    partial_df = []

    if ftype == 'netcdf':
        for f in fnames:
            try:
                ds = xr.open_dataset(f, engine='netcdf4')
            except FileNotFoundError:
                print('GSI diag file missing: %s' % f)
                continue
            # Drop the Bias_Correction_Terms variable in order to remove the 
            # Bias_Correction_Terms_arr_dim. Without this step, all observations will appear in the
            # output DataFrame 3 times!
            if 'Bias_Correction_Terms_arr_dim' in list(ds.dims.keys()):
                ds = ds.drop('Bias_Correction_Terms')
            date = ds.attrs['date_time']
            df = ds.to_dataframe()
            df['date_time'] = [date] * len(df)
            df['var'] = [f.split('_')[-2]] * len(df)
            partial_df.append(df)

    elif ftype == 'text':
        cols = ['Observation_Class', 'null1', 'Station_ID', 'null2', 'Observation_Type', 'Time', 
                'Latitude', 'Longitude', 'Pressure', 'Use_Flag', 'tmp0', 'tmp1', 'tmp2', 'tmp3', 
                'tmp4']
        for i, f in enumerate(fnames):
            try:
                df = pd.read_csv(f, delim_whitespace=True, names=cols)
            except FileNotFoundError:
                print('GSI diag file missing: %s' % f)
                continue
            df.drop(['null1', 'null2'], axis=1, inplace=True)
            # Separate out obs and O-F for winds in an effort to match the netcdf nomenclature
            for j, col in enumerate(['Observation', 'Obs_Minus_Forecast']):
                df[col] = df['tmp%d' % j]
                df['u_'+col] = df['tmp%d' % j]
                df['v_'+col] = df['tmp%d' % (j+2)]
                df.loc[df['Observation_Class'] == 'uv', col] = np.nan
                df.loc[df['Observation_Class'] != 'uv', 'u_'+col] = np.nan
                df.loc[df['Observation_Class'] != 'uv', 'v_'+col] = np.nan
            df.drop(['tmp%d' % j for j in range(5)], axis=1, inplace=True)
            if date_time[0] != None:
                df['date_time'] = date_time[i]
            partial_df.append(df)
    
    diag_out = pd.concat(partial_df)
    diag_out.reset_index(drop=True, inplace=True)

    # Add mesonet network (PRVSTG)
    if mesonet_uselist != None:
        mesonet_df = read_sfcobs_uselist(mesonet_uselist)
        mesonet_dict = {key:val for key, val in zip(mesonet_df['SID'].values, 
                                                    mesonet_df['PRVSTG'].values)}
        mesonet_sid = list(mesonet_dict.keys())

        prvstg = np.array(['MISSING']*len(diag_out), dtype='U8')
        all_sid = diag_out['Station_ID'].values
        for i in diag_out.loc[(diag_out['Observation_Type'] == 188) | 
                              (diag_out['Observation_Type'] == 195) |
                              (diag_out['Observation_Type'] == 288) |
                              (diag_out['Observation_Type'] == 295)].index:
            s = all_sid[i].decode("utf-8").strip()
            if s in mesonet_sid:
                prvstg[i] = mesonet_dict[s]
        prvstg[prvstg == 'MISSING'] = np.nan
        diag_out['PRVSTG'] = prvstg

    return diag_out


def gsi_flags_table(diag_df, field='Prep_Use_Flag'):
    """
    Returns a DataFrame detailing the number of obs that have a certain flag (e.g., Prep_Use_Flag)

    Parameters
    ----------
    diag_df : pd.DataFrame
        GSI diag DataFrame (created by read_diag)
    field : string, optional
        Field with the GSI flags

    Returns
    -------
    flag_df : pd.DataFrame
        DataFrame with the number of obs that have a certain flag

    """
    
    tmp_dict = {'Observation_Class':[], 'Observation_Type':[], field:[], 'Ob_Count':[], 
                'n_used_in_anl':[]}
    for typ in diag_df['Observation_Type'].unique():
        typ_df = diag_df.loc[diag_df['Observation_Type'] == typ]
        for flag in typ_df[field].unique():
            tmp_dict['Observation_Class'].append(typ_df['Observation_Class'].values[0])
            tmp_dict['Observation_Type'].append(typ)
            tmp_dict[field].append(flag)
            tmp_dict['Ob_Count'].append(np.sum(typ_df[field] == flag))
            tmp_dict['n_used_in_anl'].append(np.sum(typ_df.loc[typ_df[field] == flag]['Analysis_Use_Flag'] == 1))

    flag_df = pd.DataFrame(tmp_dict)
    flag_df.sort_values('Observation_Type', inplace=True)

    return flag_df


def interpolate_to_obs(diag_df, lat2d, lon2d, field2d, method='nearest'):
    """
    Interpolate a 2D field to the observation locations in a DataFrame

    Parameters
    ----------
    diag_df : pd.DataFrame
        DataFrame containing GSI diag file output
    lat2d : array
        Latitudes for the 2D field
    lon2d : array
        Longitudes for the 2D field
    field2d: array
        2D field to interpolate
    method : string, optional
        Interpolation method ('nearest' or 'linear')

    Returns
    -------
    interp_field : array
        1D array of values interpolated to the observation locations

    """

    # Convert 2D to 1D arrays
    lat1d = np.ravel(lat2d)
    lon1d = np.ravel(lon2d)
    field1d = np.ravel(field2d)

    # Create interpolation function
    if method == 'nearest':
        interp_fct = si.NearestNDInterpolator(list(zip(lon1d, lat1d)), field1d)
    elif method == 'linear':
        interp_fct = si.LinearNDInterpolator(list(zip(lon1d, lat1d)), field1d)
    else:
        print(f'interpolation method {method} is not recognized')

    # Interpolate to obs locations
    interp_field = interp_fct(diag_df['Longitude'].values - 360., diag_df['Latitude'].values)

    return interp_field


def compute_height_agl_diag(diag_df, upp_fname, interp_kw={}):
    """
    Compute height AGL for obs in a GSI diag file

    Parameters
    ----------
    diag_df : pd.DataFrame
        DataFrame containing GSI diag file output
    upp_fname : string
        UPP output file name
    interp_kw : dictionary, optional
        Keyword arguments passed to interpolate_to_obs()

    Returns
    -------
    out_df : pd.DataFrame
        Same as diag_df, but with an additional field called 'Height_AGL'

    """

    # Open UPP file
    upp_ds = xr.open_dataset(upp_fname, engine='pynio')

    # Compute height from geopotential
    upp_elev = mc.geopotential_to_height(upp_ds['HGT_P0_L1_GLC0'].values * units.m * const.g).magnitude

    # Interpolate terrain elevation to obs locations
    ob_elev = interpolate_to_obs(diag_df, upp_ds['gridlat_0'].values, upp_ds['gridlon_0'].values, 
                                 upp_elev, **interp_kw)

    # Create output DataFrame
    out_df = diag_df.copy()
    out_df['Height_AGL'] = out_df['Height'] - ob_elev

    return out_df


def test_var_f_stat(df1, df2, types=None, pcrit=0.05):
    """
    Tests whether the variances between two sets of O-B or O-A distributions are statistically 
    different using a 2-sided F test

    Parameters
    ----------
    df1 : pd.DataFrame
        First GSI diag DataFrame (created by read_diag)
    df2 : pd.DataFrame
        Second GSI diag DataFrame (created by read_diag)
    types : List, optional
        Observation types to perform the test on. Three options:
            1. None: Perform the test on each observation type in the DataFrame separately.
            2. List of Floats: Perform the test only on the listed observation types separately.
            3. List of Lists: Perform the test on each group of observation types separately. 
    pcrit : Float, optional
        P-value below which the test is considered statistically significant
        

    Returns
    -------
    test_df : pd.DataFrame
        DataFrame that contains the variances, test statistic, and p-value for each F test

    Notes
    -----
    The F test assumes that both sample are Gaussian (e.g., Wilks 2011 pg 172)
    Reference: https://www.itl.nist.gov/div898/handbook/eda/section3/eda359.htm

    """

    # Initialize dictionary
    out_dict = {}
    columns = ['ob_types', 'var1', 'var2', 'F stat', 'CDF']
    for col in columns:
        out_dict[col] = []

    # Set types if orginally None
    if types == None:
        types = list(np.intersect1d(df1['Observation_Type'].unique(), df2['Observation_Type'].unique()))

    # Perform F test
    for t in types:
        var = []
        for df in [df1, df2]:
            if type(t) == list:
                ind = np.zeros(len(df))
                for sub_t in t:
                    ind = np.logical_or(ind, df['Observation_Type'] == sub_t)
            else:
                ind = df['Observation_Type'] == t
            var.append(np.var(df['Obs_Minus_Forecast_adjusted'].loc[ind]))
        fstat = var[0] / var[1]
        out_dict['CDF'].append(ss.f.cdf(fstat, len(df1)-1, len(df2)-1))  
        out_dict['var1'].append(var[0])      
        out_dict['var2'].append(var[1])      
        out_dict['F stat'].append(fstat)
        out_dict['ob_types'].append(t)

    test_df = pd.DataFrame(out_dict)
    test_df['pcrit'] = np.ones(len(test_df)) * pcrit
    test_df['significant'] = np.logical_or(test_df['CDF'] <= (0.5*pcrit), 
                                           test_df['CDF'] >= (1-0.5*pcrit))

    return test_df


"""
End gsi_fcts.py
"""

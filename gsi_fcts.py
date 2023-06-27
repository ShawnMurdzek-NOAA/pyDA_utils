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


def read_diag(fnames):
    """
    Read a series of GSI diag netCDF4 files, concatenate, and save into a DataFrame

    Parameters
    ----------
    fnames : list of strings
        GSI diag file names
 
    Returns
    -------
    diag_out : pd.DataFrame
        GSI diag output in DataFrame format

    """

    # Read in each diag file and convert to DataFrame
    partial_df = []
    for f in fnames:
        try:
            ds = xr.open_dataset(f, engine='netcdf4')
        except FileNotFoundError:
            print('GSI diag file missing: %s' % f)
            continue
        date = ds.attrs['date_time']
        df = ds.to_dataframe()
        df['date_time'] = [date] * len(df)
        df['var'] = [f.split('_')[-2]] * len(df)
        partial_df.append(df)

    diag_out = pd.concat(partial_df)

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


"""
End gsi_fcts.py
"""

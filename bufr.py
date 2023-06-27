"""
Class to Handle BUFR Output and Input CSVs

Output BUFR CSVs are created using the prepbufr_decode_csv.f90 utility in GSI-utils.

shawn.s.murdzek@noaa.gov
Date Created: 13 October 2022
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------
 
import pandas as pd
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import scipy.interpolate as si
import collections.abc
import metpy.calc as mc
from metpy.units import units
import os
import inspect

import gsi_fcts as gsi
import meteo_util as mu


#---------------------------------------------------------------------------------------------------
# Define BUFR CSV Class
#---------------------------------------------------------------------------------------------------

class bufrCSV():
    """
    Class that handles CSV files that can easily be encoded to or decoded from BUFR files using 
    GSI-utils.

    Parameters
    ----------
    fname : string
        CSV file name
    use_all_col : boolean, optional
        Option to read in all columns

    """

    def __init__(self, fname):
   
        df = pd.read_csv(fname, dtype={'SID':str})
        df.drop(labels=df.columns[-1], axis=1, inplace=True)
  
        # Set missing values (1e11) to NaN
        self.df = df.where(df != 1e11)

        # Remove single quotes from SIDs
        sid = self.df['SID'].values
        for i in range(len(sid)):
            sid[i] = sid[i].strip("'")

        # Remove space before nmsg
        self.df.rename(columns={' nmsg':'nmsg'}, inplace=True)

        # Load metadata from JSON file
        metadata_path = '/'.join(os.path.abspath(inspect.getfile(bufrCSV)).split('/')[:-2])
        self.meta = json.load(open('%s/metadata/bufr_meta.json' % metadata_path, 'r'))

    def sample(self, fname, n=2):
        """
        Create a sample prepbufr CSV using only n lines from each unique prepbufr report type

        Parameters
        ----------
        fname : string
            Filename for sample bufr CSV file
        n : integer, optional
            Number of lines to use from each unique prepbufr report type

        Returns
        -------
        None

        """

        # Extract the first two obs for each prepbufr report type
        typ = self.df['TYP'].unique()
        df = self.df.loc[self.df['TYP'] == typ[0]].iloc[:n]
        for t in typ[1:]:
            df = pd.concat([df, self.df.loc[self.df['TYP'] == t].iloc[:n]])

        # Replace NaNs with 1e11
        df = df.fillna(1e11)

        # Reorder message numbers
        nmsg = df['nmsg'].values
        for i, msg in enumerate(df['nmsg'].unique()):
            nmsg[np.where(nmsg == msg)] = i+1
        df['nmsg'] = nmsg        

        # Overwrite aircraft and ship SIDs
        sid = df['SID'].values
        sub = df['subset'].values
        for i, s in enumerate(sub):
            if s in ['AIRCAR', 'AIRCFT', 'SFCSHP']:
                sid[i] = 'SMPL%02d' % i
        df['SID'] = sid

        # Write to CSV
        df.to_csv(fname, index=False)
   
        # Add leading blank spaces and remove the .1 and .2 from the various NUL fields
        fptr = open(fname, 'r')
        contents = fptr.readlines()
        fptr.close()

        items = contents[0].split(',')
        for i, it in enumerate(items):
            if it[:3] == 'NUL':
                items[i] = 'NUL'

        contents[0] = ','.join(items)

        fptr = open(fname, 'w')
        for l in contents:
            fptr.write(' ' + l.strip() + ',\n')
        fptr.close() 


#---------------------------------------------------------------------------------------------------
# Additional Functions
#---------------------------------------------------------------------------------------------------

def compute_Tsens(df):
    """
    Compute sensible temperature from TOB

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame

    Returns
    -------
    df : pd.DataFrame
        DataFrame with updated Tsens field

    """

    df['Tsens'] = df['TOB'].copy()
    tv_idx = np.isclose(df['tvflg'], 0)
    df.loc[tv_idx, 'Tsens'] = mu.T_from_Tv(df.loc[tv_idx, 'TOB'].values + 273.15, 
                                           df.loc[tv_idx, 'QOB'].values*1e-6) - 273.15

    return df


def compute_dewpt(df):
    """
    Compute dewpoint from T or Tv and specific humidity

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame

    Returns
    -------
    df : pd.DataFrame
        DataFrame with updated TDO field

    """

    # Add TDO column if it does not exist
    if 'TDO' not in df.columns:
        df['TDO'] = df['TOB'].copy()

    # Create temporary TOB column that only includes sensible temperature
    df = compute_Tsens(df)

    # Compute TDO
    tdo_idx = np.logical_not(np.isnan(df['TDO']))
    df.loc[tdo_idx, 'TDO'] = mc.dewpoint_from_specific_humidity(df.loc[tdo_idx, 'POB'].values * units.hPa,
                                                                df.loc[tdo_idx, 'Tsens'].values * units.degC,
                                                                df.loc[tdo_idx, 'QOB'].values * units.mg / units.kg).to('degC').magnitude

    # Drop temporary Tsens column
    df.drop(labels='Tsens', axis=1, inplace=True)

    return df


def compute_RH(df):
    """
    Compute RH from T or Tv and specific humidity

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame

    Returns
    -------
    df : pd.DataFrame
        DataFrame with RHOB field

    """

    # Create temporary TOB column that only includes sensible temperature
    df = compute_Tsens(df)

    # Compute RHOB. Don't use MetPy, as this can cause small differences when flipping back and 
    # forth between RHOB and QOB
    mix = (df['QOB'] * 1e-6) / (1. - (df['QOB'] * 1e-6))
    df['RHOB'] = 100 * mix / mu.equil_mix(df['Tsens'] + 273.15, df['POB'] * 1e2)
    #df.loc[rh_idx, 'RHOB'] = mc.relative_humidity_from_specific_humidity(df.loc[rh_idx, 'POB'].values * units.hPa,
    #                                                            df.loc[rh_idx, 'Tsens'].values * units.degC,
    #                                                            df.loc[rh_idx, 'QOB'].values * units.mg / units.kg).to('percent').magnitude

    # Drop temporary TOB column
    df.drop(labels='Tsens', axis=1, inplace=True)

    return df


def compute_wspd_wdir(df):
    """
    Compute wind speed (m/s) and wind direction (deg)

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame

    Returns
    -------
    df : pd.DataFrame
        DataFrame with WSPD and WDIR fields

    """

    df['WSPD'] = mc.wind_speed(df['UOB'].values * units.m / units.s,
                               df['VOB'].values * units.m / units.s).to(units.m / units.s).magnitude
    df['WDIR'] = mc.wind_direction(df['UOB'].values * units.m / units.s,
                                   df['VOB'].values * units.m / units.s).magnitude

    return df


def plot_obs_locs(x, y, fig=None, nrows=1, ncols=1, axnum=1, proj=ccrs.PlateCarree(), 
                  borders=True, **kwargs):
    """
    Plot observation locations using CartoPy.

    Parameters
    ----------
    x, y : float
        Observation locations (deg E, deg N)
    fig : matplotlib.pyplot.figure, optional
        Figure object to add axes to
    nrows, ncols, axnum : integer, optional
        Number of rows/columns of axes, and axes number for this plot
    proj : optional
        CartoPy map projection
    borders : boolean, optional
        Option to add country borders
    **kwargs : optional
        Other keyword arguments passed to matplotlib.pyplot.plot()

    Returns
    -------
    ax : matplotlib.axes
        Matplotlib.axes instance

    """   
  
    if fig == None:
        fig = plt.figure()

    ax = fig.add_subplot(nrows, ncols, axnum, projection=proj)

    # Add map features
    ax.coastlines('50m')
    if borders:
        ax.add_feature(cfeature.BORDERS)

    ax.plot(x, y, transform=proj, **kwargs)

    return ax


def df_to_csv(df, fname):
    """
    Write a DataFrame to a BUFR CSV file

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame
    fname : string
        Filename for bufr CSV file

    Returns
    -------
    None

    """

    # Replace NaNs with 1e11
    df = df.fillna(1e11)

    # Reorder message numbers
    nmsg = df['nmsg'].values
    for i, msg in enumerate(df['nmsg'].unique()):
        nmsg[np.where(nmsg == msg)] = i+1
    df['nmsg'] = nmsg        

    # Place SIDs in quotes (otherwise an error occurs when converting back to BUFR format)
    sid = df['SID'].values
    for i in range(len(sid)):
        if sid[i][0] != "'":
            sid[i] = "'%s'" % sid[i]
    df['SID'] = sid

    # Write to CSV
    df.to_csv(fname, index=False)
   
    # Add leading blank spaces and remove the .1 and .2 from the various NUL fields
    fptr = open(fname, 'r')
    contents = fptr.readlines()
    fptr.close()

    items = contents[0].split(',')
    for i, it in enumerate(items):
        if it[:3] == 'NUL':
            items[i] = 'NUL'

    contents[0] = ','.join(items)

    fptr = open(fname, 'w')
    for l in contents:
        fptr.write(' ' + l.strip() + ',\n')
    fptr.close() 


def create_uncorr_obs_err(err_num, stdev):
    """
    Create uncorrelated observation errors using a Gaussian distribution with a mean of 0

    Parameters
    ----------
    err_num : integer
        Number of errors to create
    stdev : float or array
        Observation error standard deviation (either a single value or array with size err_num)

    Returns
    -------
    error : array
        Uncorrelated random observation errors

    """

    if isinstance(stdev, (collections.abc.Sequence, np.ndarray)):
        error = np.zeros(err_num)
        for i, s in enumerate(stdev):
            error[i] = np.random.normal(scale=s)
    else:
        error = np.random.normal(scale=stdev, size=err_num)

    return error


def create_corr_obs_err(ob_df, stdev, auto_dim, partition_dim=None, auto_reg_parm=0.5, 
                        min_d=0.01667):
    """
    Create correlated observation errors using an AR1 process

    Parameters
    ----------
    ob_df : DataFrame
        DataFrame containing decoded BUFR observations
    stdev : float or array
        Observation error standard deviation (either a single value or one per entry in ob_df)
    auto_dim : string
        Dimension along which to have autocorrelation. typically 'POB' or 'DHR'
    partition_dim : string, optional
        Dimension used for partitioning. All entries with the same value along this dimension use 
        the same AR1 process, and a new process is created when the value along this dimension 
        changes
    auto_reg_parm : float, optional
        Autoregression parameter (see Notes)
    min_d : float, optional
        Minimum distance allowed between successive points when computing the autocorrelation (see
        Notes). This parameter is necessary b/c the ob spacing in 'POB' or 'DHR' is not constant.

    Returns
    -------
    error : array
        Autocorrelated random observation errors

    Notes
    -----
    Errors are computed using the following equation: 
    
    error = N(0, stdev) + auto_reg_parm * error(n-1) / d,

    where N(0, stdev) is a random draw from a Gaussian distribution with mean 0 and a prescribed
    standard deviation (stdev), error(n-1) is the previous error value, and d is a modified 
    distance between the two obs. d is defined as follows:

    d = 1                for distances <= min_d
    d = distance / min_d for distances > min_d  

    """

    error = np.zeros(len(ob_df))
    ob_df.reset_index(inplace=True, drop=True)

    # For POB, sort based on descending values
    if auto_dim == 'POB':
        ascending = False
    else:
        ascending = True

    # Determine if the stdev is a scalar or a list/array
    if isinstance(stdev, (collections.abc.Sequence, np.ndarray)):
        is_stdev_array = True
    else:
        is_stdev_array = False

    # Loop over each station ID, then over each sorted index
    for sid in ob_df['SID'].unique():
       single_station = ob_df.loc[ob_df['SID'] == sid].copy()
       if partition_dim != None:
           partitioned_obs = []
           partition_vals = single_station[partition_dim].unique()
           for v in partition_vals:
               partitioned_obs.append(single_station.loc[single_station[partition_dim] == v])  
       else:
           partitioned_obs = [single_station]
       for obs in partitioned_obs:
           idx = obs.sort_values(auto_dim, ascending=ascending).index
           dist = np.abs(obs.loc[idx[1:], auto_dim].values - 
                         obs.loc[idx[:-1], auto_dim].values)
           dist[dist <= min_d] = min_d
           dist = dist / min_d
           if is_stdev_array:
               error[idx[0]] = np.random.normal(scale=stdev[idx[0]])
               for j1, j2, d in zip(idx[:-1], idx[1:], dist):
                  error[j2] = np.random.normal(scale=stdev[j2]) + (auto_reg_parm * error[j1] / d)
           else:
               error[idx[0]] = np.random.normal(scale=stdev)
               for j1, j2, d in zip(idx[:-1], idx[1:], dist):
                   error[j2] = np.random.normal(scale=stdev) + (auto_reg_parm * error[j1] / d)
   
    return error


def add_obs_err(df, errtable, ob_typ='all', correlated=None, auto_reg_parm=0.5, min_d=0.01667,
                partition_dim=None, verbose=True):
    """
    Add random Gaussian errors (correlated or uncorrelated) to observations based on error standard 
    deviations in errtable

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame
    errtable : string
        Name of errtable text file
    ob_typ: 'all' or list, optional
        List of observation types to apply errors to 
    correlated : None, 'POB', or 'DHR'; optional
        Option for correlating observation errors. Can be uncorrelated, correlated in the vertical,
        or correlated in time
    auto_reg_parm : float, optional
        Autoregressive parameter for correlated errors. Correlated errors are computed by assuming
        an AR1 process
    min_d : float, optional
        Minimum allowed distance between successive obs
    partition_dim : string, optional
        Dimension used for partitioning. All entries with the same value along this dimension use 
        the same AR1 process, and a new process is created when the value along this dimension 
        changes
    verbose : boolean, optional
        Option for verbose output 

    Returns
    -------
    out_df : pd.DataFrame
        DataFrame of observations with random errors added

    """
    
    # Make copy of DataFrame
    out_df = df.copy()

    # Determine list of all observation types
    if ob_typ == 'all':
        ob_typ = np.int32(out_df['TYP'].unique())

    # Read in error table file 
    etable = gsi.read_errtable(errtable)
    eprs = etable[100]['prs'].values

    # Convert specific humidities to relative humidities in BUFR CSV
    out_df = compute_RH(out_df)
    out_df['RHOB'] = 0.1 * out_df['RHOB']

    # Convert surface pressures from Pa to hPa in BUFR CSV
    out_df['PRSS'] = out_df['PRSS'] * 1e-2

    # Loop over each observation type
    for t in ob_typ:
        if verbose:
            print()
            print('TYP = %d' % t)
        for ob, err in zip(['TOB', 'RHOB', 'UOB', 'VOB', 'PRSS', 'PWO', 'PMO'],
                           ['Terr', 'RHerr', 'UVerr', 'UVerr', 'PSerr', 'PWerr', 'PSerr']):
        
            if verbose:    
                print(ob)
            # Check to see if errors are defined
            if np.all(np.isnan(etable[t][err].values)):
                if verbose:
                    print('no errors for ob_typ = %d, ob = %s' % (t, ob))
                continue

            # Determine indices where errors are not NaN
            eind = np.where(np.logical_not(np.isnan(etable[t][err])))[0]
            
            # Determine indices in out_df for this ob type
            # All pressures are set to NaN for ob type 153, so don't use POB to determine oind for 
            # those obs
            if t == 153:
                oind = out_df['TYP'] == t
            else:
                oind = ((out_df['TYP'] == t) & (out_df['POB'] <= eprs[eind[0]]) & 
                        (out_df['POB'] >= eprs[eind[-1]]))

            # Determine if errors vary with pressure. If so, create a function for interpolation
            # Then add observation errors
            if len(np.unique(etable[t].loc[eind, err])) == 1:
                stdev = etable[t].loc[eind[0], err]
            else:
                stdev_fct_p = si.interp1d(eprs[eind], etable[t].loc[eind, err])
                stdev = stdev_fct_p(out_df.loc[oind, 'POB'])

            # Compute errors
            if correlated == None:
                error = create_uncorr_obs_err(np.sum(oind), stdev)
            else:
                error = create_corr_obs_err(out_df.loc[oind].copy(), stdev, correlated, 
                                            auto_reg_parm=auto_reg_parm, min_d=min_d)

            out_df.loc[oind, ob] = out_df.loc[oind, ob] + error

    # Set relative humidities > 100% to 100%
    out_df.loc[out_df['RHOB'] > 10, 'RHOB'] = 10.

    # Prevent negative values for certain obs
    out_df.loc[out_df['RHOB'] < 0, 'RHOB'] = 0.
    out_df.loc[out_df['PWO'] < 0, 'PWO'] = 0.

    # Compute specific humidity from relative humidity
    out_df = compute_Tsens(out_df)
    mix = 0.1 * out_df['RHOB'] * mu.equil_mix(out_df['Tsens'] + 273.15, out_df['POB'] * 1e2)
    out_df['QOB'] = 1e6 * (mix / (1. + mix))
    out_df.drop(labels='RHOB', axis=1, inplace=True)    
    out_df.drop(labels='Tsens', axis=1, inplace=True)    
   
    # Recompute dewpoint
    out_df = compute_dewpt(out_df)
 
    # Convert surface pressure back to Pa
    out_df['PRSS'] = out_df['PRSS'] * 1e2
    
    return out_df


def match_bufr_prec(df, prec_csv='bufr_precision.csv'):
    """
    Round observations so the precision matches what is typically found in a BUFR file

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame
    prec_csv : string
        File name (within the py_scripts/metadata directory) that contains the number of decimal
        points and minimum values for each variable

    Returns
    -------
    df : pd.DataFrame
        DataFrame with obs rounded to the appropriate decimal place

    """

    # Open precision CSV file
    metadata_path = '/'.join(os.path.abspath(inspect.getfile(match_bufr_prec)).split('/')[:-2])
    prec_df = pd.read_csv('%s/metadata/%s' % (metadata_path, prec_csv))

    # Compute wind speed (in kts) and direction (in deg)
    df = compute_wspd_wdir(df)
    df['WSPD'] = (df['WSPD'].values * units.m / units.s).to('kt').magnitude    

    for t in np.unique(df['TYP']):
        cond = (df['TYP'] == int(t))
        for v in ['TOB', 'QOB', 'POB', 'WSPD', 'WDIR', 'ELV', 'ZOB', 'PWO']:

            # Apply minimum threshold for each variable
            thres = prec_df.loc[prec_df['TYP'] == t, '%s_min' % v].values[0]
            if not np.isnan(thres):
                df.loc[cond & (df[v] <= thres), v] = 0
            
            # Round results
            ndec = int(prec_df.loc[prec_df['TYP'] == t, '%s_ndec' % v].values[0])
            if v in ['QOB', 'POB', 'WSPD', 'WDIR', 'PWO']:
                df.loc[cond & (df[v] <= (10**(-ndec))), v] = 0
            df.loc[cond, v] = np.around(df.loc[cond, v], decimals=ndec)       
   
        # Convert WSPD and WDIR back to UOB and VOB
        tmp = mc.wind_components(df.loc[cond, 'WSPD'].values * units('kt'), 
                                 df.loc[cond, 'WDIR'].values * units('deg'))
        df.loc[cond, 'UOB'] = tmp[0].to(units.m / units.s).magnitude
        df.loc[cond, 'VOB'] = tmp[1].to(units.m / units.s).magnitude

        # Round UOB and VOB
        for v in ['UOB', 'VOB']:
            ndec = int(prec_df.loc[prec_df['TYP'] == t, '%s_ndec' % v].values[0])
            df.loc[cond & (df[v] <= (10**(-ndec))) & (df[v] >= (-10**(-ndec))), v] = 0
            df.loc[cond, v] = np.around(df.loc[cond, v], decimals=ndec)

    # Drop WSPD and WDIR
    df.drop(labels=['WSPD', 'WDIR'], axis=1, inplace=True)

    return df


def RH_check(df):
    """
    Reduce QOB values so that RH stays below 100%

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame

    Returns
    -------
    df : pd.DataFrame
        DataFrame with obs rounded to the appropriate decimal place

    """

    # Convert specific humidities to relative humidities
    q = out_df['QOB'] * 1e-6
    mix = q  / (1. - q)
    RH = (mix / mu.equil_mix(out_df['TOB'] + 273.15, out_df['POB'] * 1e2))  

    RH[RH > 1] = 0
    
    # Convert back to specific humidity
    mix = RH * mu.equil_mix(out_df['TOB'] + 273.15, out_df['POB'] * 1e2)
    out_df['QOB'] = 1e6 * (mix / (1. + mix))

    return df


"""
End bufr.py
"""

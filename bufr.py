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
import warnings
import datetime as dt

import pyDA_utils.gsi_fcts as gsi
import pyDA_utils.meteo_util as mu


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
   
        df = pd.read_csv(fname, dtype={'SID':str, 'PRVSTG':str, 'SPRVSTG':str})
        df.drop(labels=df.columns[-1], axis=1, inplace=True)

        # Pandas sets empty strings to NaNs, but we want to retain these empty strings
        for field in ['SID', 'PRVSTG', 'SPRVSTG']:
            df.loc[df[field] != df[field], field] = "''"
  
        # Set missing values (1e11) to NaN
        self.df = df.where(np.logical_and(df != 1e11, df != '100000000000.0000'))

        # Remove space before nmsg
        self.df.rename(columns={' nmsg':'nmsg'}, inplace=True)

        # Load metadata from JSON file
        metadata_path = '/'.join(os.path.abspath(inspect.getfile(bufrCSV)).split('/')[:-1])
        self.meta = json.load(open('%s/metadata/bufr_meta.json' % metadata_path, 'r'))


    def select_obtypes(self, ob_types):
        """
        Subset the prepbufr CSV so that it only contains observations included in the ob_types list

        Parameters
        ----------
        ob_types : list of integers
            Observation types (3-digit numbers) to include. All other observations are discarded.

        """
       
        all_types = self.df['TYP'].unique()
        ob_types = np.array(ob_types, dtype=np.float64)
        for typ in all_types:
            if typ not in ob_types:
                self.df = self.df.loc[~np.isclose(self.df['TYP'].values, typ)]
    

    def select_SIDs(self, sids):
        """
        Subset the prepbufr CSV so that it only contains station IDs included in the sids list

        Parameters
        ----------
        sids : list
            Station IDs to retain
        
        """

        cond = np.zeros(len(self.df))
        for s in sids:
            cond = cond + (self.df['SID'] == s)
        
        self.df = self.df.loc[cond > 0, :]
    

    def select_dhr(self, DHR):
        """
        Subset the prepbufr CSV so that it only contains observations that are closest for a 
        particular valid time

        Parameters
        ----------
        DHR : float
            Observation valid time

        """

        all_typ = np.unique(self.df['TYP'])
        keep_idx = []
        for t in all_typ:
            t_cond = self.df['TYP'] == t
            all_sid = np.unique(self.df['SID'].loc[t_cond])
            for s in all_sid:
                s_cond = self.df['SID'] == s
                min_dhr = self.df.loc[np.abs(self.df['DHR'].loc[t_cond & s_cond] - DHR).idxmin(), 'DHR']
                keep_idx = keep_idx + list(self.df.loc[t_cond & s_cond & np.isclose(self.df['DHR'], min_dhr)].index)
                
        self.df = self.df.loc[keep_idx]


    def select_latlon(self, minlat, minlon, maxlat, maxlon):
        """
        Subset the prepbufr CSV so that it only contains observations within a given (lat, lon) box

        Parameters
        ----------
        minlat : float
            Minimum latitude (deg N)
        minlon : float
            Minimum longitude (deg E, range 0 to 360)
        maxlat : float
            Maximum latitude (deg N)
        maxlon : float
            Maximum longitude (deg E, range 0 to 360)

        """

        self.df = self.df.loc[(self.df['XOB'] >= minlon) &
                              (self.df['XOB'] <= maxlon) &
                              (self.df['YOB'] >= minlat) &
                              (self.df['YOB'] <= maxlat), :]
                

    def match_types(self, typ1, typ2, match_fields=['SID', 'XOB', 'YOB'], nearest_field='DHR', 
                    copy_fields=[]):
        """
        Match all typ1 observations to a typ2 observation. Ideally used to match thermodynamic to
        kinematic obs

        Parameters
        ----------
        typ1 : string
            Observation type to loop over
        typ2 : string
            Observation type that typ1 is matched to (NOTE: not every typ2 ob is guaranteed to be 
            matched to a typ1 ob)
        match_fields : list of strings, optional
            Fields that must match exactly between typ1 and typ2
        nearest_fields : string, optional
            In the event that multiple typ2 obs match after filtering based on the match_fields, 
            the ob with the value of nearest_field closed to typ1 is matched
        copy_fields : list of strings, optional
            Fields to copy from typ2 to typ1

        Returns
        -------
        None. self.df has a new field: match. The same value of "match" indicates that two 
        observations have been matched together
            
        """

        # Extract required fields from DataFrame
        all_fields = match_fields + [nearest_field, 'TYP'] + copy_fields
        fields = {}
        for f in all_fields:
            fields[f] = self.df[f].values
        nobs = len(fields[all_fields[0]])
        fields['idx'] = np.arange(nobs, dtype=int)

        # Determine indices for the first ob type
        ob_idx_1 = np.where(fields['TYP'] == typ1)[0]

        # Create a match field (the same value indicates that observations are matched)
        # Note that a "-1" indicates that there is no match
        match = np.zeros(nobs, dtype=int)
        m = 1
        for i in ob_idx_1:
            cond = (fields['TYP'] == typ2)
            for f in match_fields:
                cond = cond * (fields[f] == fields[f][i])
            if np.sum(cond) == 0:
                match[i] = -1
            else:
                if np.sum(cond) == 1:
                    t2_idx = np.where(cond)[0][0]
                else:
                    red_idx = np.argmin(np.abs(fields[nearest_field][i] - fields[nearest_field][cond]))
                    t2_idx = fields['idx'][cond][red_idx]
                for c in copy_fields:
                    fields[c][i] = fields[c][t2_idx]
                match[i] = m
                match[t2_idx] = m
                m = m + 1

        # Add match field to DataFrame
        self.df['match'] = match
        for c in copy_fields:
            self.df[c] = fields[c]


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


    def ob_hist(self, var, ax=None, hist_kw={}, plot_kw={}):
        """
        Plot all observation values as a histogram

        Parameters
        ----------
        var : string
            Variable to plot (e.g., TOB, QOB)
        ax : matplotlib.axes, optional
            Axes to plot histogram on. Set to None to create a new figure
        hist_kw : dictionary, optional
            Keyword arguments to pass to numpy.histogram
        plot_kw : dictionary, optional
            Keyword arguments to pass to matplotlib.pyplot.plot

        Returns
        -------
        ax : matplotlib.axes
            Axes with plotted data

        """

        # Create histogram counts
        cts, bin_edges = np.histogram(self.df[var].values, **hist_kw)

        # Plot data
        if ax == None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.set_xlabel(var, size=12)

        ax.plot(0.5*(bin_edges[:-1] + bin_edges[1:]), cts, **plot_kw)

        return ax


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


def plot_obs(bufr_df, colorcode=None, fig=None, nrows=1, ncols=1, axnum=1, 
             proj=ccrs.LambertConformal(), borders=True, scale='50m', **kwargs):
    """
    Plot observation locations using CartoPy.

    Parameters
    ----------
    bufr_df : pd.DataFrame
        Input BUFR data as a DataFrame
    colorcode : string, optional
        Color-code observations by a certain value. Set to None if not used.
    fig : matplotlib.pyplot.figure, optional
        Figure object to add axes to
    nrows, ncols, axnum : integer, optional
        Number of rows/columns of axes, and axes number for this plot
    proj : optional
        CartoPy map projection
    borders : boolean, optional
        Option to add country borders
    scale : string, optional
        Scale for the map features (e.g., coastlines)
    **kwargs : optional
        Other keyword arguments passed to matplotlib.pyplot.scatter()

    Returns
    -------
    ax : matplotlib.axes
        Matplotlib.axes instance

    """   
  
    if fig == None:
        fig = plt.figure()

    ax = fig.add_subplot(nrows, ncols, axnum, projection=proj)

    # Add map features
    ax.coastlines(scale)
    if borders:
        borders = cfeature.NaturalEarthFeature(category='cultural',
                                               scale=scale,
                                               facecolor='none',
                                               name='admin_1_states_provinces')
        ax.add_feature(borders)

    # Plot data
    if colorcode:
        plot_df = bufr_df.loc[~np.isnan(bufr_df[colorcode])].copy()
        cax = ax.scatter(plot_df['XOB'], plot_df['YOB'], c=plot_df[colorcode], 
                         transform=ccrs.PlateCarree(), **kwargs)
        cbar = plt.colorbar(cax, ax=ax)
        cbar.set_label(colorcode, size=14)
    else:
        plot_df = bufr_df.copy()
        ax.scatter(plot_df['XOB'], plot_df['YOB'], transform=ccrs.PlateCarree(), **kwargs)

    return ax


def df_to_csv(df, fname, quotes=True):
    """
    Write a DataFrame to a BUFR CSV file

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame
    fname : string
        Filename for bufr CSV file
    quotes : boolean, optional
        Option to place output strings in quotes (necessary if converting back to BUFR)

    Returns
    -------
    None

    """

    # Reorder message numbers
    nmsg = df['nmsg'].values
    for i, msg in enumerate(df['nmsg'].unique()):
        nmsg[np.where(nmsg == msg)] = i+1
    df['nmsg'] = nmsg        

    # Place strings in quotes (otherwise an error occurs when converting back to BUFR format)
    # The `tmp == tmp` line checks for NaNs
    if quotes:
        for field in ['SID', 'PRVSTG', 'SPRVSTG']:
            tmp = df[field].values
            new = np.empty([len(tmp)], dtype=object)
            for j in range(len(tmp)):
                if (tmp[j] == tmp[j]):
                    if (tmp[j][0] != "'"):
                        new[j] = "'%s'" % tmp[j]
                    else:
                        new[j] = tmp[j]
                else:
                    new[j] = "'100000000000.0000'"
            df[field] = new
    
    # Replace NaNs with 1e11
    df = df.fillna(1e11)

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
                        min_d=0.01667, verbose=0):
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
    verbose : integer, optional
        Verbosity level. Higher numbers correspond to more output

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

    # Extract necessary fields from DataFrame
    all_idx = ob_df.index.values
    all_sid = ob_df['SID'].values
    all_auto_dim = ob_df[auto_dim].values
    if partition_dim != None:
        all_partition_dim = ob_df[partition_dim].values

    # For POB, sort based on descending values
    if auto_dim == 'POB':
        ascending = False
    else:
        ascending = True

    # Determine if the stdev is a scalar or a list/array
    is_stdev_array = isinstance(stdev, (collections.abc.Sequence, np.ndarray))

    # Loop over each station ID, then over each sorted index
    for sid in ob_df['SID'].unique():
        if verbose > 0: print(f'{sid} ({dt.datetime.now()})') 
        if verbose > 1: print(f'  determining tmp_idx1 ({dt.datetime.now()})')
        if partition_dim != None:
            partition_dim_vals = np.unique(all_partition_dim[all_sid == sid])
            tmp_idx1 = [np.where(np.logical_and(all_sid == sid, all_partition_dim == val))[0] for val in partition_dim_vals]
        else:
            tmp_idx1 = np.where(all_sid == sid)

        for i1 in tmp_idx1:
            if verbose > 1: print(f'  determining tmp_idx2 ({dt.datetime.now()})')
            sub_auto_dim = all_auto_dim[i1]
            tmp_idx2 = np.argsort(sub_auto_dim)
            if not ascending:
                tmp_idx2 = tmp_idx2[::-1]
            if verbose > 1: print(f'  computing d ({dt.datetime.now()})')
            dist = np.abs(sub_auto_dim[tmp_idx2[1:]] - sub_auto_dim[tmp_idx2[:-1]])
            dist[dist <= min_d] = min_d
            dist = dist / min_d
            if verbose > 1: print(f'  computing error ({dt.datetime.now()})')
            idx = i1[tmp_idx2]
            if is_stdev_array:
                error[idx[0]] = np.random.normal(scale=stdev[idx[0]])
                for j1, j2, d in zip(idx[:-1], idx[1:], dist):
                    error[j2] = np.random.normal(scale=stdev[j2]) + (auto_reg_parm * error[j1] / d)
            else:
                error[idx[0]] = np.random.normal(scale=stdev)
                for j1, j2, d in zip(idx[:-1], idx[1:], dist):
                    error[j2] = np.random.normal(scale=stdev) + (auto_reg_parm * error[j1] / d)

    return error


def add_obs_err(df, std_errtable, mean_errtable=None, ob_typ='all', correlated=None, 
                auto_reg_parm=0.5, min_d=0.01667, partition_dim=None, verbose=True):
    """
    Add random Gaussian errors (correlated or uncorrelated) to observations based on error standard 
    deviations in var_errtable

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the same format as a BUFR DataFrame
    std_errtable : string
        Name of errtable text file that contains error standard deviations
    mean_errtable : string, optional
        Name of errtable text file that contains error means (i.e., bias). 
        If set to None, use mean = 0.
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
    if verbose: print(f'Copying DataFrame ({dt.datetime.now()})...')
    out_df = df.copy()

    # Determine list of all observation types
    if verbose: print(f'Determining obs types ({dt.datetime.now()})...')
    if ob_typ == 'all':
        ob_typ = np.int32(out_df['TYP'].unique())

    # Read in error table file(s) 
    if verbose: print(f'Reading error table ({dt.datetime.now()})...')
    etable = gsi.read_errtable(std_errtable)
    eprs = etable[100]['prs'].values
    if mean_errtable != None:
        mean_etable = gsi.read_errtable(mean_errtable)
        mean_eprs = mean_etable[100]['prs'].values
        for key in mean_errtable.keys():
            mean_errtable[key].fillna(0, inplace=True)

    # Convert specific humidities to relative humidities in BUFR CSV
    if verbose: print(f'Computing RH ({dt.datetime.now()})...')
    out_df = compute_RH(out_df)
    out_df['RHOB'] = 0.1 * out_df['RHOB']

    # Convert surface pressures from Pa to hPa in BUFR CSV
    if verbose: print(f'Converting PRSS to hPa ({dt.datetime.now()})...')
    out_df['PRSS'] = out_df['PRSS'] * 1e-2

    # Loop over each observation type
    for t in ob_typ:
        if verbose:
            print()
            print('TYP = %d' % t)
        for ob, err in zip(['TOB', 'RHOB', 'UOB', 'VOB', 'PRSS', 'PWO', 'PMO'],
                           ['Terr', 'RHerr', 'UVerr', 'UVerr', 'PSerr', 'PWerr', 'PSerr']):
        
            if verbose: print(f'  {ob} ({dt.datetime.now()})')
            # Check to see if errors are defined
            if (np.all(np.isnan(etable[t][err].values)) or 
                 np.all(np.isclose(etable[t][err].values, 0))):
                if verbose:  print('  no errors for ob_typ = %d, ob = %s' % (t, ob))
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
                if t == 153:
                    print('Type = 153 error variances vary with P, but type 153 obs DO NOT have' +
                          ' corresponding POBs. To prevent all type = 153 obs from being NaN, the' +
                          ' first entry in the errtable will be used for the error variances')
                    stdev = etable[t].loc[eind[0], err]

            # Interpolate error means, if specified
            if mean_errtable != None:
                mean_fct_p = si.interp1d(mean_eprs, mean_etable[t].loc[:, err])
                mean = mean_fct_p(out_df[oind, 'POB'])
            else:
                mean = 0

            # Compute errors
            if verbose: print(f'    computing errors ({dt.datetime.now()})')
            if correlated == None:
                error = mean + create_uncorr_obs_err(np.sum(oind), stdev)
            else:
                error = mean + create_corr_obs_err(out_df.loc[oind].copy(), stdev, correlated, 
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
    metadata_path = '/'.join(os.path.abspath(inspect.getfile(match_bufr_prec)).split('/')[:-1])
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

    out_df = df.copy()

    # Convert specific humidities to relative humidities
    q = out_df['QOB'] * 1e-6
    mix = q  / (1. - q)
    RH = (mix / mu.equil_mix(out_df['TOB'] + 273.15, out_df['POB'] * 1e2))  

    RH[RH > 1] = 1
    
    # Convert back to specific humidity
    mix = RH * mu.equil_mix(out_df['TOB'] + 273.15, out_df['POB'] * 1e2)
    out_df['QOB'] = 1e6 * (mix / (1. + mix))

    return out_df


def combine_bufr(df_list):
    """
    Combine several BUFR CSV DataFrames into one

    Parameters
    ----------
    df_list : list of pd.DataFrame
        List of DataFrames containing BUFR output

    Returns
    -------
    combined : pd.DataFrame
        Combined DataFrame

    """

    for i in range(1, len(df_list)):
        df_list[i]['nmsg'] = df_list[i]['nmsg'] + df_list[i-1]['nmsg'].max()
    combined = pd.concat(df_list)
    combined.reset_index(inplace=True, drop=True)

    return combined 


def compute_ceil(df, use_typ=[187], no_ceil=2e4):
    """
    Compute cloud ceilings using cloud amount (CLAM) and cloud base (HOCB)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing BUFR output
    use_typ : list of integers, optional
        List of observation types (3-digit codes) to compute cloud ceilings for
    no_ceil : float, optional
        Fill value for no ceiling detected (which is different from a missing ceiling!)

    Returns
    -------
    ceil : array
        Ceiling field. NaNs indicate missing obs and 2e4 indicates no ceiling detected.

    Notes
    -----
    This code is loosely based on the nonvariational cloud analysis

    THIS CODE HAS NOT BEEN TESTED, USE AT YOUR OWN RISK
    """

    warnings.warn("Warning: Function to compute cloud ceilings is not thoroughly tested." +
                  " Consider using the CEILING field instead")

    # Initialize ceil field
    ceil = np.ones(len(df)) * np.nan

    # Extract some arrays
    clam = df['CLAM'].values
    hocb = df['HOCB'].values
    sid = df['SID'].values
    dhr = df['DHR'].values
    typ = df['TYP'].values

    # Create a reduced CLAM array to aid in computing ceilings
    clam_red = np.ones(clam.size, dtype=np.int32) * -1
    clam_red[((clam >= -0.1) & (clam <= 3.1)) | np.isclose(clam, 11) | np.isclose(clam, 13)] = 0
    clam_red[((clam >= 3.9) & (clam <= 9.1)) | np.isclose(clam, 12)] = 1

    # Determine ceilings
    for t in use_typ:
        sid_for_t = sid[typ == t]
        dhr_for_t = dhr[typ == t]
        for s in np.unique(sid_for_t):
            for d in np.unique(dhr_for_t[s == sid_for_t]):
                idx = np.where((s == sid) & np.isclose(d, dhr) & (t == typ))[0]
                if np.all(clam_red[idx] < 0):
                    # CLAM in (10, 14, 15, or another number)
                    # Missing value
                    continue
                elif np.any(clam_red[idx] == 1):
                    # CLAM in (4, 5, 6, 7, 8, 9, 12)
                    # Ceiling detected
                    ceil[idx[0]] = np.nanmin(hocb[idx][clam_red[idx] == 1])
                else:
                    # CLAM is in (0, 1, 2, 3, 11, 13)
                    # No ceiling detected
                    ceil[idx[0]] = no_ceil

    return ceil


"""
End bufr.py
"""

"""
Cloud Data Assimilation Forward Operator

Passed Arguments
----------------
    sys.argv[1] : BUFR CSV observation file name
    sys.argv[2] : UPP model output
    sys.argv[3] : Option to create plots (default: No plots. Set to 1 to make plots)

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
import pyproj
import scipy.spatial as ss
import scipy.interpolate as si
import scipy.ndimage as sn
import matplotlib.pyplot as plt

from pyDA_utils import bufr
import pyDA_utils.upp_postprocess as uppp
import pyDA_utils.cloud_DA_forward_operator_viz as cfov


#---------------------------------------------------------------------------------------------------
# Classes and Functions
#---------------------------------------------------------------------------------------------------

class sfc_cld_forward_operator():
    """
    Forward operator that interpolates model cloud fraction to ceilometer obs

    Parameters
    ----------
    ob_df : pd.DataFrame
        BUFR ceilometer cloud obs used by the forward operator
    model_ds : xr.Dataset
        Model background
    debug : integer, optional
        Debugging level (higher numbers print more output)

    """

    def __init__(self, ob_df, model_ds, debug=0):
        self.ob_df = ob_df
        self.model_ds = model_ds
        self.debug = debug

        # Define primary data structure used by this class
        self.data = {}
        keys1d = ['idx', 'lat', 'lon', 'TYP', 'SID'] 
        keys2d = ['DHR', 'CLAM', 'HOCB']
        for k in keys1d + keys2d:
            self.data[k] = []

        i = 0
        for t in np.unique(self.ob_df['TYP']):
            t_cond = self.ob_df['TYP'] == t
            for s in np.unique(self.ob_df['SID'].loc[t_cond]):
                subset = self.ob_df.loc[t_cond & (self.ob_df['SID'] == s), :].copy()
                subset.sort_values('HOCB', inplace=True)
                self.data['idx'].append(i)
                self.data['lat'].append(subset['YOB'].values[0])
                self.data['lon'].append(subset['XOB'].values[0] - 360.)
                self.data['TYP'].append(t)
                self.data['SID'].append(s)
                self.data['DHR'].append(subset['DHR'].values)
                self.data['CLAM'].append(subset['CLAM'].values)
                self.data['HOCB'].append(subset['HOCB'].values)
                i = i + 1

        # Change lists to arrays
        for k in keys1d:
            self.data[k] = np.array(self.data[k])


    def decode_ob_clam(self):
        """
        Decode ceilometer cloud amount field (CLAM)
        See the BUFR table here: https://www.emc.ncep.noaa.gov/mmb/data_processing/table_20.htm#0-20-011

        Parameters
        ----------
        None

        Returns
        -------
        None. Adds 'ob_cld_amt' and 'ob_cld_precision' fields to self.data

        """
        self.data['ob_cld_amt'] = []
        self.data['ob_cld_precision'] = []
        for i in self.data['idx']:
            nob = len(self.data['CLAM'][i])
            self.data['ob_cld_amt'].append(np.zeros(nob))
            self.data['ob_cld_precision'].append(np.zeros(nob))
            for j in range(nob):
                if self.data['CLAM'][i][j] < 9:
                    self.data['ob_cld_amt'][i][j] = 100 * (self.data['CLAM'][i][j] / 8.)
                    self.data['ob_cld_precision'][i][j] = 12.5
                elif np.isclose( self.data['CLAM'][i][j], 11):
                    self.data['ob_cld_amt'][i][j] = 37.5
                    self.data['ob_cld_precision'][i][j] = 25.
                elif np.isclose( self.data['CLAM'][i][j], 12):
                    self.data['ob_cld_amt'][i][j] = 75.
                    self.data['ob_cld_precision'][i][j] = 0.25
                elif np.isclose( self.data['CLAM'][i][j], 13):
                    self.data['ob_cld_amt'][i][j] = 12.5
                    self.data['ob_cld_precision'][i][j] = 25.


    def interp_model_col_to_ob(self, method='nearest', proj_str='+proj=lcc +lat_0=39 +lon_0=-96 +lat_1=33 +lat_2=45',
                               fields=['TCDC_P0_L105_GLC0', 'height_agl']):
        """
        Wrapper method for interpolation of model vertical columns to ceilometer (lat, lon) locations

        Parameters
        ----------
        method : string, optional
            Interpolation method. Only supported method currently is 'nearest'
        proj_str : string, optional
            Map projection string for pyproj
        fields : list of strings, optional
            Model fields to interpolate
        
        Returns
        -------
        None. Adds fields 'model_col_<field>' to self.data

        """

        self._apply_map_projection(proj_str=proj_str)
        self._compute_model_height_agl()

        if self.debug > 0:
            print('sfc_cld_forward_operator: Starting interpolation of model output columns...')
            start = dt.datetime.now()

        if method == 'nearest':
            self._model_nearest_interp_col(fields=fields)
        else:
            print(f'method {method} is not supported')
        
        if self.debug > 0:
            print('sfc_cld_forward_operator: Finished interpolating model output columns ' +
                  f'(elapsed time = {(dt.datetime.now() - start).total_seconds()} s)')
 
    
    def _apply_map_projection(self, proj_str='+proj=lcc +lat_0=39 +lon_0=-96 +lat_1=33 +lat_2=45'):
        """
        Apply map projection to model and obs (lat, lon) locations. Interpolation should ideally
        be performed in the map projection space.

        Parameters
        ----------
        proj_str : string, optional
            Map projection string for pyproj
        
        Returns
        -------
        None. Adds fields 'x_proj' and 'y_proj' to self.data

        """

        self.proj_str = proj_str
        self.proj = pyproj.Proj(proj_str)

        self.data['x_proj'], self.data['y_proj'] = self.proj(self.data['lon'], self.data['lat'])

        xtmp, ytmp = self.proj(self.model_ds['gridlon_0'].values, self.model_ds['gridlat_0'].values)
        for data, name in zip([xtmp, ytmp], ['x_proj', 'y_proj']):
            self.model_ds[name] = xr.DataArray(data=data, 
                                               dims=self.model_ds['gridlon_0'].dims,
                                               coords=self.model_ds['gridlon_0'].coords,
                                               attrs={'units':'m'})
    

    def _compute_model_height_agl(self):
        """
        Compute model heights AGL (from gpm MSL)

        Parameters
        ----------
        None

        Returns
        -------
        None. Adds the field 'height_agl' to self.model_ds

        """

        self.model_ds['height_agl'] = xr.DataArray(data=uppp.convert_gpm_msl_to_m_agl(self.model_ds, 'HGT_P0_L105_GLC0'),
                                                   dims=self.model_ds['HGT_P0_L105_GLC0'].dims,
                                                   coords=self.model_ds['HGT_P0_L105_GLC0'].coords,
                                                   attrs={'long_name':'height AGL', 'units':'m'})
    

    def _model_nearest_interp_col(self, fields=['TCDC_P0_L105_GLC0', 'height_agl']):
        """
        Use nearest neighbor interpolation to interpolate model columns to obs (lat, lon) locations

        Parameters
        ----------
        fields : list of strings, optional
            Fields to interpolat
        
        Returns
        -------
        None. Adds the fields 'model_col_<field>' to self.data

        """

        fields_model = {}
        for f in fields:
            self.data[f'model_col_{f}'] = []
            fields_model[f] = self.model_ds[f].values

        xmodel = self.model_ds['x_proj'].values
        ymodel = self.model_ds['y_proj'].values

        model_KDTree = ss.KDTree(np.array([xmodel.ravel(), ymodel.ravel()]).T)
        _, idx1d = model_KDTree.query(np.array([self.data['x_proj'], self.data['y_proj']]).T)
        for idx in idx1d:
            i, j = np.unravel_index(idx, xmodel.shape)
            for f in fields:
                self.data[f'model_col_{f}'].append(fields_model[f][:, i, j])
    

    def impose_hgt_limits(self, min_hgt=10, max_hgt=3658, hgt_field='model_col_height_agl',
                          fields=['model_col_height_agl', 'model_col_TCDC_P0_L105_GLC0']):
        """
        Impose min and max height limits to either the model or obs

        Parameters
        ----------
        min_hgt : float, optional
            Minimum height to search for clouds (m AGL)
        max_hgt : float, optional
            Maximum height to search for clouds (m AGL)
        hgt_field : string, optional
            Field containing heights AGL in the model
        fields : list of strings, optional
            Model fields to truncate based on min_hgt and max_hgt
        
        Returns
        -------
        None

        """

        for i in self.data['idx']:

            # Remove model (or obs) outside of [min_hgt, max_hgt]
            # Should correspond to the min/max ceilometer vertical range
            cond_hgt = np.logical_and(self.data[hgt_field][i] >= min_hgt, self.data[hgt_field][i] <= max_hgt)
            for f in fields:
                self.data[f][i] = self.data[f][i][cond_hgt]
    

    def impose_min_cld_frac(self, min_cld_frac=5, field='model_col_TCDC_P0_L105_GLC0'):
        """
        Impose a minimum cloud fraction to the model cloud fraction field

        Parameters
        ----------
        min_cld_frac : integer, optional
            Model cloud fraction below which cloud fractions are set to 0 (%)
        field : string, optional
            Model field containing cloud fractions
        
        Returns
        -------
        None

        """

        for i in self.data['idx']:

            # Set cloud fractions below min_cld_frac to 0
            # Should correspond to minimum cloud fraction that ceilometer can observe
            cond_0 = self.data[field][i] <= min_cld_frac
            self.data[field][i][cond_0] = 0


    def clean_obs(self):
        """
        Remove entries that no longer have obs associated with them

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        keep_idx = []
        for i in self.data['idx']:
            if len(self.data['HOCB'][i]) > 0:
                keep_idx.append(i)
        
        for k in self.data.keys():
            self.data[k] = [self.data[k][j] for j in keep_idx]
        
        self.data['idx'] = np.arange(len(keep_idx))

    
    def add_clear_obs_model(self, vert_roi=250):
        """
        Add clear obs where there are model clouds but no observed clouds

        Parameters
        ----------
        vert_roi : float, optional
            Vertical radius of influence (m)
        
        Returns
        -------
        None

        """

        print("sfc_cld_forward_operator.add_clear_obs_model() is deprecated!" + 
              "Try using sfc_cld_forward_operator.add_clear_obs_model() instead.")

        # Check whether ob_cld_amt is defined
        try: 
            self.data['ob_cld_amt']
            amt_field = 'ob_cld_amt'
        except KeyError:
            amt_field = 'CLAM'

        for i in self.data['idx']:
            bin_cld = self.data['model_col_TCDC_P0_L105_GLC0'][i] > 0.1

            if ~np.isclose(self.data['CLAM'][i][0], 0):
                # At least one cloudy ob
                for j, (zob, amtob) in enumerate(zip(self.data['HOCB'][i], self.data['CLAM'][i])):
                    # Ignore model clouds within vert_roi of cloud ob
                    bin_cld[np.logical_and(self.data['model_col_height_agl'][i] >= (zob - vert_roi),
                                           self.data['model_col_height_agl'][i] <= (zob + vert_roi))] = 0
                    if (np.isclose(amtob, 8) or j == 3):
                        # No obs above an overcast ob
                        # Also cannot have > 3 obs, so if this is the highest ob, ignore any other model clouds
                        bin_cld[self.data['model_col_height_agl'][i] > zob] = 0

            # Add clear obs
            if np.sum(bin_cld) > 0:
                label_cld, nlabel = sn.label(bin_cld)
                for j in range(1, nlabel+1):
                    self.data['HOCB'][i] = np.concatenate([self.data['HOCB'][i], 
                                                           np.array([np.mean(self.data['model_col_height_agl'][i][label_cld == j])])])
                    self.data[amt_field][i] = np.concatenate([self.data[amt_field][i], np.array([0])])


    def add_clear_obs(self, clr_ob_locs=np.arange(250, 3500, 500), terminate_clr_col=[8]):
        """
        Add clear obs at regular intervals where ceilometers do not report clouds

        Parameters
        ----------
        clr_ob_locs : np.array, optional
            Vertical locations for clear obs (m AGL)
        terminate_clr_col : list, optional
            CLAM values used to terminate column of clear obs (i.e., do not add clear obs above the 
            height of one of the CLAM values)
        
        Returns
        -------
        None

        """

        # Check whether ob_cld_amt is defined
        try: 
            self.data['ob_cld_amt']
            amt_field = 'ob_cld_amt'
        except KeyError:
            amt_field = 'CLAM'

        if self.debug > 0:
            print()
            print('Inside sfc_cld_forward_operator.add_clear_obs()...')
            print(f'amt_field = {amt_field}')
            print(f'clr_ob_locs = {clr_ob_locs}')

        for i in self.data['idx']:
            if np.isclose(self.data[amt_field][i][0], 0):
                # Clear ceilometer ob
                self.data['HOCB'][i] = clr_ob_locs
                self.data[amt_field][i] = np.zeros(len(clr_ob_locs))
            
            else:
                # At least one cloudy ob
                mask = np.ones(len(clr_ob_locs), dtype=bool)
                for j, (zob, amtob) in enumerate(zip(self.data['HOCB'][i], self.data['CLAM'][i])):
                    # Remove the closest clear ob to each cloudy ob
                    mask[np.argmin(np.abs(clr_ob_locs - zob))] = 0
                    if ((amtob in terminate_clr_col) or j == 3):
                        # No obs above an overcast ob
                        # Also cannot have > 3 obs, so if this is the highest ob, 
                        # do not add clear obs above this ob
                        mask[clr_ob_locs >= zob] = 0
                self.data['HOCB'][i] = np.concatenate([self.data['HOCB'][i], clr_ob_locs[mask]])
                self.data[amt_field][i] = np.concatenate([self.data[amt_field][i], np.zeros(len(clr_ob_locs[mask]))])
                idx_sort = np.argsort(self.data['HOCB'][i])
                self.data['HOCB'][i] = self.data['HOCB'][i][idx_sort]
                self.data[amt_field][i] = self.data[amt_field][i][idx_sort]

                if ((self.debug > 1) and (i < 5)):
                    print(i)
                    print(f'mask = {mask}')
                    print(f'clear ob hgts = {clr_ob_locs[mask]}')
                    print(f"data['HOCB'][i] = {self.data['HOCB'][i]}")
                    print(f'data[amt_field][i] = {self.data[amt_field][i]}')


    def interp_model_to_obs(self, method='nearest', match_precision=True):
        """
        Perform interpolation within a model column to the observed cloud locations

        Parameters
        ----------
        method : string, optional
            Interpolation method. Passed to si.interp1d
        match_precision : boolean, optional
            Option to have the interpolated cloud fraction match the precision of the obs
        
        Returns
        -------
        None. Adds the field 'hofx' to self.data

        """

        self.data['hofx'] = []
        for i in self.data['idx']:
            interp_fct = si.interp1d(self.data['model_col_height_agl'][i], 
                                     self.data['model_col_TCDC_P0_L105_GLC0'][i], 
                                     kind=method,
                                     bounds_error=False,
                                     fill_value="extrapolate")  # Not sure if "extrapolate" will have undesirable results...
            self.data['hofx'].append(interp_fct(self.data['HOCB'][i]))

            # This block of code is rather ad hoc and offers a lot of opportunity for tuning
            if match_precision:
                for j in range(len(self.data['hofx'][i])):
                    if ((self.data['hofx'][i][j] >= (self.data['ob_cld_amt'][i][j] - self.data['ob_cld_precision'][i][j])) and
                        (self.data['hofx'][i][j] <= (self.data['ob_cld_amt'][i][j] + self.data['ob_cld_precision'][i][j])) and
                        (self.data['hofx'][i][j] >= 1) and (self.data['hofx'][i][j] <= 99) and
                        (self.data['ob_cld_amt'][i][j] >= 1) and (self.data['ob_cld_amt'][i][j] <= 99)):
                         self.data['hofx'][i][j] = self.data['ob_cld_amt'][i][j]
    

    def compute_OmB(self):
        """
        Compute O-B

        Parameters
        ----------
        None

        Returns
        -------
        None. Adds the field 'OmB' to self.data

        """

        self.data['OmB'] = []
        for i in self.data['idx']:
            self.data['OmB'].append(self.data['ob_cld_amt'][i] - self.data['hofx'][i])

    
    def flatten1d(self, field):
        """
        Create a 1D array with all values in self.data[field]
        
        Parameters
        ----------
        field : string
            Field from self.data to flatten

        Returns
        -------
        np.array
            Flattened data

        """
        out_list = []
        for val in self.data[field]:
            out_list = out_list + list(val)
        return np.array(out_list)


    def compute_global_RMS(self, field='OmB'):
        """
        Compute the RMS value for the input field across the entire domain

        Parameters
        ----------
        field : string, optional
            Field from self.data to compute the root-mean-square of
        
        Returns
        -------
        rms : float
            Root-mean-square of the input field

        """

        field1d = self.flatten1d(field)
        rms = np.sqrt(np.mean(np.array(field1d)**2))

        return rms


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
                          (~np.isclose(bufr_df['CLAM'], 9)) &
                          (~np.isclose(bufr_df['CLAM'], 10)) &
                          (~np.isclose(bufr_df['CLAM'], 14)) &
                          (~np.isclose(bufr_df['CLAM'], 15)) &
                          (~(~np.isclose(bufr_df['CLAM'], 0) & np.isnan(bufr_df['HOCB'])))]
    bufr_df.reset_index(inplace=True, drop=True)

    return bufr_df


def ceilometer_hofx_driver(cld_ob_df, model_ds, debug=0, verbose=1, 
                           interp_col_kw={}, hgt_lim_kw={}, min_frac_kw={}, clr_ob_kw={},
                           interp_z_kw={}):
    """
    Main driver for the ceilometer cloud forward operator

    Parameters
    ----------
    cld_ob_df : pd.DataFrame
        Ceilometer cloud obs
    model_ds : xr.Dataset
        Background model data
    debug : integer, optional
        Debug level. Increase for more debugging output
    verbose : integer, optional
        Verbosity. Increase for more general output
    interp_col_kw : dictionary, optional
        Keyword arguments passed to sfc_cld_forward_operator.interp_model_col_to_ob()
    hgt_lim_kw : dictionary, optional
        Keyword arguments passed to sfc_cld_forward_operator.impose_hgt_limits()
    min_frac_kw : dictionary, optional
        Keyword arguments passed to sfc_cld_forward_operator.impose_min_cld_frac()
    clr_ob_kw : dictionary, optional
        Keyword arguments passed to sfc_cld_forward_operator.add_clear_obs()
    interp_z_kw : dictionary, optional
        Keyword arguments passed to sfc_cld_forward_operator.interp_model_to_obs()
    
    Returns
    -------
    cld_hofx : sfc_cld_forward_operator
        Ceilometer cloud forward operator object
        
    """

    if verbose > 0: print('Creating ceilometer forward operator object')
    cld_hofx = sfc_cld_forward_operator(cld_ob_df, model_ds, debug=debug)

    if verbose > 0: print('Interpolating model columns to obs locations...')
    cld_hofx.interp_model_col_to_ob(**interp_col_kw)

    if verbose > 0: print('Imposing height limits and min cld fraction...')
    cld_hofx.impose_hgt_limits(hgt_field='model_col_height_agl',
                               fields=['model_col_height_agl', 'model_col_TCDC_P0_L105_GLC0'],
                               **hgt_lim_kw)
    cld_hofx.impose_min_cld_frac(**min_frac_kw)

    # Set clear obs to have HOCB = 50 m so they don't get removed
    for i in cld_hofx.data['idx']:
        if np.isclose(cld_hofx.data['CLAM'][i][0], 0):
            cld_hofx.data['HOCB'][i][0] = 50
    cld_hofx.impose_hgt_limits(hgt_field='HOCB',
                               fields=['CLAM', 'HOCB'],
                               **hgt_lim_kw)
    cld_hofx.clean_obs()
    for i in cld_hofx.data['idx']:
        if np.isclose(cld_hofx.data['CLAM'][i][0], 0):
            cld_hofx.data['HOCB'][i][0] = np.nan
    
    if verbose > 0: print('Adding clear obs and interpolating model clouds in column to ob heights...')
    cld_hofx.add_clear_obs(**clr_ob_kw)
    cld_hofx.decode_ob_clam()
    cld_hofx.interp_model_to_obs(**interp_z_kw)

    return cld_hofx
    

if __name__ == '__main__':

    start = dt.datetime.now()
    print(f'start time: {start.strftime("%Y%m%d %H:%M:%S")}')
    print()

    print('reading in BUFR obs...')
    bufr_obj = bufr.bufrCSV(sys.argv[1])

    print('reading in model UPP output...')
    model_ds = xr.open_dataset(sys.argv[2], engine='pynio')

    print('Identifying cloud obs...')
    cld_ob_df = find_bufr_cloud_obs(bufr_obj)

    print('Running ceilometer forward operator...')
    cld_hofx = ceilometer_hofx_driver(cld_ob_df, model_ds, debug=0, hgt_lim_kw={'max_hgt':3500})

    print()
    cld_hofx.compute_OmB()
    print(f"O-B RMSD = {cld_hofx.compute_global_RMS(field='OmB')}")

    if len(sys.argv) > 3:
        if sys.argv[3]:
            print()
            print('Making plots...')
            cld_hofx_viz = cfov.sfc_cld_forward_operator_viz(cld_hofx)
            cld_hofx_viz.scatterplot()
            plt.savefig('cld_amt_scatterplot.png')
            cld_hofx_viz.hist(plot_param={'field':'OmB'})
            plt.savefig('OmB_hist.png')

    print()
    print(f'Done! Elapsed time = {(dt.datetime.now() - start).total_seconds()} s')


"""
End cloud_DA_forward_operator.py
"""
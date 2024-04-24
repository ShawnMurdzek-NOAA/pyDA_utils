"""
Class for Computing Superobs from BUFR CSV Files

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr
import scipy.interpolate as si
import metpy.interpolate as mi

import pyDA_utils.bufr as bufr
import pyDA_utils.map_proj as mp


#---------------------------------------------------------------------------------------------------
# Superob Class
#---------------------------------------------------------------------------------------------------

class superobPB(bufr.bufrCSV):
    """
    Class that handles superobbing prepbufr CSV files

    Parameters
    ----------
    fname : string
        BUFR CSV file name
    debug : integer, optional
        Debug level. Higher number indicates more output printed to the screen
    map_proj : function, optional
        Function used for the map projection to map the observations to the superob grid. Only 
        used when using the 'grid' grouping option.
    map_proj_kw : dictionary, optional
        Keyword arguments for map_proj

    """

    def __init__(self, fname, debug=0,
                 map_proj=mp.ll_to_xy_lc, map_proj_kw={'dx':3, 'knowni':899, 'knownj':529}):
        super().__init__(fname)
        self.full_df = self.df.copy()
        self.debug = debug
        self.map_proj = map_proj
        self.map_proj_kw = map_proj_kw


    def create_superobs(self, obtypes=[136], grouping='temporal', grouping_kw={}, reduction_kw={}, 
                        rh_check=True):
        """
        Main driver to create superobs

        Parameters
        ----------
        obtypes : list of integers, optional
            Observation types to create superobs for
        grouping : string, optional
            Grouping strategy used for superobs (options: 'temporal' and 'grid')
        grouping_kw : dictionary, optional
            Keyword argument passed to the grouping method
        reduction_kw : dictionary, optional
            Keyword arguments passed to the reduction method
        rh_check : boolean, optional
            Option to adjust relative humidities > 100% down to 100% after creating superobs

        Returns
        -------
        superobs : pd.DataFrame
            DataFrame containing superobs

        Notes
        -----
        Superobbing is performed in two general steps:
        
        1) Grouping: Assign raw observations to a superob group. All raw observations within a given
            group will be combined into a single superob. Superob groups are listed in the 
            'superob_groups' column in self.df. Note that the grouping methods do not return anything,
            instead, they add the 'superob_groups' column to self.df. An example of a superob group is 
            all observations within a 30-s window.

        2) Reduction: Reduce all raw observations within a superob group into a single observation.
            An example of a reduction is taking the mean of all the raw observations within a group.
        
        """ 

        # Only retain obs that will be part of the superob
        self.select_obtypes(obtypes)

        # Perform superob grouping
        self.assign_superob(grouping, grouping_kw=grouping_kw)

        # Perform superob reduction
        superobs = self.reduction_superob(**reduction_kw)

        # Clean up
        superobs.reset_index(inplace=True, drop=True)

        # Ensure that RH stays below 100%
        superobs = bufr.RH_check(superobs)

        return superobs
    

    def assign_superob(self, grouping, grouping_kw={}):
        """
        Wrapper for superob assignment (grouping) functions

        Parameters
        ----------
        grouping : string
            Grouping strategy. Options: 'temporal' or 'grid'.
        grouping_kw : dictionary, optional
            Keyword arguments passed to the grouping method

        Returns
        -------
        None
        
        """

        if grouping == 'temporal':
            self.grouping_temporal(**grouping_kw)
        elif grouping == 'grid':
            self.grouping_grid(**grouping_kw)
        else:
            print(f'grouping method {grouping} is not a valid option')

    
    def grouping_temporal(self, window=60):
        """
        Create superob groups based on a temporal window (in sec)

        Parameters
        ----------
        window : float, optional
            Temporal window used to group raw observations (s)
            
        Returns
        -------
        None
        
        """

        # Prep
        self.df['superob_groups'] = np.zeros(len(self.df), dtype=int)
        self.df.sort_values(['SID', 'DHR'], inplace=True)
        sids = np.unique(self.df['SID'])
        window_dhr = window / 3600.
        min_dhr = np.amin(self.df['DHR'].values)
        max_dhr = np.amax(self.df['DHR'].values)

        # Assign superob group numbers
        group = 0
        for window_min in np.arange(min_dhr, max_dhr - (0.5*window_dhr), window_dhr):
            dhr_cond = np.logical_and(self.df['DHR'] >= window_min, self.df['DHR'] < (window_min + window_dhr))
            for s in sids:
                self.df.loc[(self.df['SID'] == s) & dhr_cond, 'superob_groups'] = group
                group = group + 1


    def grouping_grid(self, grid_fname='/work2/noaa/wrfruc/murdzek/src/pyDA_utils/tests/data/RRFS_grid_max.nc',
                      grid_field_names={'x':'lon', 'y':'lat', 'sfc':'HGT_SFC', 'z':'HGT_AGL'},
                      check_proj=True,
                      subtract_360_lon_grid=True,
                      interp_kw={}):
        """
        Create superob groups based on an input grid

        Parameters
        ----------
        grid_fname : string, optional
            NetCDF file containing the grid used for creating superob groups
        grid_field_names : dictionary, optional
            Names of the x, y, x, and surface height fields in grid_fname
        check_proj : boolean, optional
            Option to check whether the map projection is appropriate for the grid_fname
        subtract_360_lon_grid : boolean, optional
            Option to subtract 360 deg from the longitude coordinates in grid_fname
        interp_kw : dictionary, optional
            Keyword arguments passed to the method that interpolates the surface height field
            to the observation locations

        Returns
        -------
        None
        
        """

        # Prep
        grid_ds = xr.open_dataset(grid_fname)
        ny, nx = grid_ds[grid_field_names['x']].shape
        nz = grid_ds[grid_field_names['z']].size

        # First test to ensure that the map projection is appropriate
        if check_proj:
            tol = 0.6 / self.map_proj_kw['dx']
            if subtract_360_lon_grid:
                lon = grid_ds[grid_field_names['x']].values - 360
            else:
                lon = grid_ds[grid_field_names['x']].values
            x_rmse, y_rmse = mp.rmse_map_proj(grid_ds[grid_field_names['y']].values,
                                              lon,
                                              proj=self.map_proj,
                                              proj_kw=self.map_proj_kw)
            if (x_rmse > tol) or (y_rmse > tol):
                print('map projection is not appropriate for this grid')
                print(f'tolerance = {tol}')
                print(f'X RMSE = {x_rmse}')
                print(f'Y RMSE = {y_rmse}')

        # Perform map projection on obs
        self.df = self.map_proj_obs(self.df)

        # Determine height AGL of obs
        hgt_sfc = grid_ds[grid_field_names['sfc']].values
        x1d_grid = np.arange(grid_ds[grid_field_names['x']].shape[1])
        y1d_grid = np.arange(grid_ds[grid_field_names['x']].shape[0])
        self.interp_gridded_field_obs('SFC', (y1d_grid, x1d_grid), hgt_sfc, interp_kw=interp_kw)
        obs_hgt_agl = self.df['ZOB'].values - self.df['SFC'].values

        # Assign superob groups
        xgroup = np.floor(self.df['XMP'])
        ygroup = np.floor(self.df['YMP']) 
        zgroup = np.zeros(len(self.df))
        for i in range(nz):
            zgroup = zgroup + (obs_hgt_agl > grid_ds[grid_field_names['z']].values[nz-i-1])
        self.df['superob_groups'] = xgroup + (ygroup*nx) + (zgroup*nx*ny)      

  
    def map_proj_obs(self, df):
        """
        Perform map projection on observation locations

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing observations (i.e., data read from a BUFR CSV file)

        Returns
        -------
        df : pd.DataFrame
            Same as the input DataFrame, but with extra columns (XMP, YMP) that give the 
            observation coordinates based on the map projection
            
        """

        xob, yob = self.map_proj(df['YOB'], df['XOB'] - 360, **self.map_proj_kw)
        df.loc[:, 'XMP'] = xob
        df.loc[:, 'YMP'] = yob

        return df


    def interp_gridded_field_obs(self, outfield, grid_pts, grid_vals, interp_kw={}):
        """
        Interpolate a gridded field to observation locations

        Parameters
        ----------
        outfield : string
            Column name to save interpolated output to
        grid_pts : tuple of arrays
            Coordinate of gridded input, in format (x, y)
        grid_vals : array
             Values corresponding to grid_pts
        interp_kw : dictionary, optional
            Keyword arguments passed to RegularGridInterpolator

        Returns
        -------
        None
        
        """

        # Create interpolation object
        interp = si.RegularGridInterpolator(grid_pts, grid_vals, **interp_kw)

        # Apply interpolation object
        obs_pts = np.array([[y, x] for x, y in zip(self.df['XMP'], self.df['YMP'])])
        self.df[outfield] = interp(obs_pts)


    def reduction_superob(self, var_dict={'TOB':{'method':'mean', 'qm_kw':{'field':'TQM', 'thres':2}, 'reduction_kw':{}}}):
        """
        Wrapper for superob reduction functions

        Parameters
        ----------
        var_dict : dictionary, optional
            Reduction options for each variable being reduced. Keys are the variables, and the values
            are the reduction options. Options must include the following:
                'method' : Method used for reduction. Options: 'mean', 'hor_cressman', 'vert_cressman'
                'qm_kw' : Keywords for quality control
                'reduction_kw' : Additional keywords passed to the reduction method

        Returns
        -------
        superobs : pd.DataFrame
            DataFrame containing superobs
        
        """

        # Create superob DataFrame for results
        superobs = self.df.drop_duplicates('superob_groups').copy()

        # Check to see if options are defined for coordinates (XOB, YOB, ZOB, DHR)
        # Use mean if not defined
        all_keys = list(var_dict.keys())
        superob_keys = ['XOB', 'YOB', 'ZOB', 'DHR']
        for key in superob_keys:
            if key in all_keys:
                all_keys.remove(key)
            else:
                var_dict[key] = {'method':'mean', 'reduction_kw':{}}
        superob_keys = superob_keys + all_keys

        # Perform superob reduction
        for field in superob_keys:

            # Perform quality control
            if 'qm_kw' in var_dict[field].keys():
                qc_df = self.qc_obs(**var_dict[field]['qm_kw'])
            else:
                qc_df = self.df.copy()
   
            # Perform reduction
            if var_dict[field]['method'] == 'mean':
                superobs.loc[:, field] = self.reduction_mean(qc_df, superobs, field, **var_dict[field]['reduction_kw'])
            elif var_dict[field]['method'] == 'hor_cressman':
                superobs.loc[:, field] = self.reduction_hor_cressman(qc_df, superobs, field, **var_dict[field]['reduction_kw'])
            elif var_dict[field]['method'] == 'vert_cressman':
                superobs.loc[:, field] = self.reduction_vert_cressman(qc_df, superobs, field, **var_dict[field]['reduction_kw'])
            else:
                print(f"reduction method {var_dict[field]['method']} is not a valid option")
        
        return superobs


    def qc_obs(self, field='TQM', thres=2):
        """
        Quality control observations

        Parameters
        ----------
        field : string, optional
            Quality mark field
        thres : integer, optional
            Quality mark threshold. All observations with quality marks above this threshold are removed

        Returns
        -------
        qc_df : pd.DataFrame
            DataFrame only containing observations that passed QC
            
        """

        qc_df = self.df.copy()

        # Apply quality marks
        qc_df = qc_df.loc[qc_df[field] <= thres]

        return qc_df


    def reduction_mean(self, qc_df, superobs_in, field):
        """
        Superob reduction using a regular mean

        Parameters
        ----------
        qc_df : pd.DataFrame
            Input DataFrame containing quality-controlled raw observations with assigned superob groups
        superobs_in : pd.DataFrame
            DataFrame containing fields that have already been superobbed
        field : string
            Field to create superobs for (e.g., 'TOB')

        Returns
        -------
        superobs : array
            Superobs for the given field
            
        """

        superob_groups = superobs_in['superob_groups'].values
        superobs = np.zeros(len(superob_groups)) * np.nan
        for i, g in enumerate(superob_groups):
            values = qc_df.loc[qc_df['superob_groups'] == g, field].values
            if len(values) > 0:
                superobs[i] = np.nanmean(values)

        return superobs
    
    
    def reduction_hor_cressman(self, qc_df, superob_in, field, R=1, use_metpy=False, min_neighbor=3):
        """
        Superob reduction in horizontal using a Cressman successive corrections method

        Parameters
        ----------
        qc_df : pd.DataFrame
            Input DataFrame containing quality-controlled raw observations with assigned superob groups
        superobs_in : pd.DataFrame
            DataFrame containing fields that have already been superobbed. Necessary because the Cressman
            average needs the coordinates of the superob groups
        field : string
            Field to create superobs for (e.g., 'TOB')
        R : integer, optional
            Maximum search radius in the horizontal. Uses the same units as XMP and YMP
        use_metpy : boolean, optional
            Option to use the MetPy Cressman implementation
        min_neighbor : integer, optional
            Minimum number of neighbors to create a superob. NaNs are assigned to groups with too few
            neighbors. 

        Returns
        -------
        superobs : array
            Superobs for the given field
          
        """

        # Define R2
        R2 = R*R

        # Perform map projection on superob coordinates
        superob_in = self.map_proj_obs(superob_in)
        superob_groups = superob_in['superob_groups'].values
        superob_x = superob_in['XMP'].values
        superob_y = superob_in['YMP'].values

        # Create superobs
        superobs = np.zeros(len(superob_groups)) * np.nan
        for i, g in enumerate(superob_groups):
            subset_df = qc_df.loc[qc_df['superob_groups'] == g].copy()
            if len(subset_df) >= min_neighbor:
                raw_vals = subset_df[field].values
                if use_metpy:
                    ob_pts = np.array([[x, y] for x, y in zip(subset_df['XMP'].values, subset_df['YMP'].values)])
                    superob_pts = np.array([[superob_x[i], superob_y[i]]])
                    superobs[i] = mi.interpolate_to_points(ob_pts, raw_vals, superob_pts,
                                                           interp_type='cressman',
                                                           minimum_neighbors=min_neighbor, search_radius=R)[0]
                else:
                    d2 = ((subset_df['XMP'] - superob_x[i])**2 + 
                          (subset_df['YMP'] - superob_y[i])**2).values
                    d2[d2 > R2] = np.nan
                    wgts = (R2 - d2) / (R2 + d2)
                    if np.nansum(wgts) == 0:
                        superobs[i] = np.nanmean(raw_vals)
                    else:
                        superobs[i] = np.nansum(wgts * raw_vals) / np.nansum(wgts)

        return superobs


    def reduction_vert_cressman(self, qc_df, superob_in, field, R=100, use_metpy=False, min_neighbor=3):
        """
        Superob reduction in vertical using a Cressman successive corrections method
        
        Parameters
        ----------
        qc_df : pd.DataFrame
            Input DataFrame containing quality-controlled raw observations with assigned superob groups
        superobs_in : pd.DataFrame
            DataFrame containing fields that have already been superobbed. Necessary because the Cressman
            average needs the coordinates of the superob groups
        field : string
            Field to create superobs for (e.g., 'TOB')
        R : integer or 'max', optional
            Maximum search radius in the horizontal. Uses the same units as XMP and YMP. 'Max' sets
            R to be the maximum distance between any raw observation height and the superob group height.
        use_metpy : boolean, optional
            Option to use the MetPy Cressman implementation
        min_neighbor : integer, optional
            Minimum number of neighbors to create a superob. NaNs are assigned to groups with too few
            neighbors. It is recommended that a value > 1 is used, as using min_neighbor = 1 with 
            R = 'max' results in a divide by zero warning because R2 = d2 = 0

        Returns
        -------
        superobs : array
            Superobs for the given field
          
        """

        # Define R2
        if R != 'max':
            R2 = R*R

        # Define superob ZOB
        superob_groups = superob_in['superob_groups'].values
        superob_z = superob_in['ZOB'].values

        # Create superobs
        superobs = np.zeros(len(superob_groups)) * np.nan
        for i, g in enumerate(superob_groups):
            subset_df = qc_df.loc[qc_df['superob_groups'] == g].copy()
            if len(subset_df) >= min_neighbor:
                raw_vals = subset_df[field].values
                d2 = (subset_df['ZOB'].values - superob_z[i])**2
                if R == 'max':
                    R2 = np.amax(d2)
                else:
                    d2[d2 > R2] = np.nan
                if use_metpy:
                    ob_pts = np.array([[z, 0] for z in subset_df['ZOB'].values])
                    superob_pts = np.array([[superob_z[i], 0]])
                    superobs[i] = mi.interpolate_to_points(ob_pts, raw_vals, superob_pts, interp_type='cressman',
                                                           minimum_neighbors=min_neighbor, search_radius=np.sqrt(R2))[0]
                else:
                    wgts = (R2 - d2) / (R2 + d2)
                    if np.nansum(wgts) == 0:
                        superobs[i] = np.nanmean(raw_vals)
                    else:
                        superobs[i] = np.nansum(wgts * raw_vals) / np.nansum(wgts)

        return superobs


"""
End superob_prepbufr.py
"""

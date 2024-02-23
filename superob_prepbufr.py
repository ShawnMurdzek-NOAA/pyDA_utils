"""
Class for Computing Superobs from BUFR CSV Files

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr

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

    """

    def __init__(self, fname):
        super().__init__(fname)
        self.full_df = self.df.copy()


    def create_superobs(self, obtypes=[136], grouping='temporal', grouping_kw={}, reduction_kw={})
        """
        Main driver to create superobs
        """ 

        # Only retain obs that will be part of the superob
        self.select_obtypes(obtypes)

        # Perform superob grouping
        self.assign_superob(grouping, grouping_kw=grouping_kw)

        # Perform superob reduction
        superobs = self.reduction_superob(**reduction_kw)

        # Clean up
        superobs.reset_index(inplace=True, drop=True)

        return superobs
    

    def assign_superob(self, grouping, grouping_kw={}):
        """
        Wrapper for superob assignment functions
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
        for window_min in np.arange(min_dhr, max_dhr - (0.5*window_dhr), window_dhr)
            dhr_cond = np.logical_and(self.df['DHR'] >= window_min, self.df['DHR'] < (window_min + window_dhr))
            for s in sids:
                self.df.loc[(self.df['SID'] == s) & dhr_cond, 'superob_groups'] = group
                group = group + 1


    def grouping_grid(self, grid_fname='/work2/noaa/wrfruc/murdzek/src/pyDA_utils/tests/data/RRFS_grid_max.nc',
                      grid_field_names={'x':'lon', 'y':'lat', 'z':'HGT_AGL'},
                      map_proj=mp.ll_to_xy_lc,
                      map_proj_kw={'dx':3, 'knowni':899, 'knownj':529},
                      check_proj=True,
                      subtract_360_lon_grid=True):
        """
        Create superob groups based on the model grid
        """

        # Prep
        grid_ds = xr.open_dataset(grid_fname)
        ny, nx = grid_ds[grid_field_names['x']].shape
        nz = grid_ds[grid_field_names['z']].size

        # First test to ensure that the map projection is appropriate
        if check_proj:
            tol = 0.2
            if subtract_360_lon_grid:
                lon = grid_ds[grid_field_names['x']].values - 360
            else:
                lon = grid_ds[grid_field_names['x']].values
            x_rmse, y_rmse = mp.rmse_map_proj(grid_ds[grid_field_names['y']],
                                              lon,
                                              proj=map_proj,
                                              proj_kw=map_proj_kw)
            if (x_rmse > tol) or (y_rmse > tol):
                print('map projection is not appropriate for this grid')
                print(f'X RMSE = {x_rmse}')
                print(f'Y RMSE = {y_rmse}')

        # Perform map projection on obs
        xob, yob = map_proj(self.df['YOB'], self.df['XOB'] - 360, **map_proj_kw)

        # Assign superob groups
        xgroup = np.floor(xob)
        ygroup = np.floor(yob) 
        zgroup = np.zeros(len(xgroup))
        for i in range(nz):
            zgroup = zgroup + (self.df['ZOB'] > grid_ds[grid_field_names['z']][nz-i-1])
        self.df['superob_groups'] = xgroup + (ygroup*nx) + (zgroup*nx*ny)      


    def reduction_superob(self, var_dict={'TOB':{'method':'mean', 'qm_kw':{'field':'TQM', 'thres':2}, 'reduction_kw':{}}}):
        """
        Wrapper for superob reduction functions
        """

        # Create superob DataFrame for results
        superobs = self.df.drop_duplicates('superob_groups')

        # Perform superob reduction
        for field in var_dict.keys():
            qc_df = self.qc_obs(**var_dict[field]['qm_kw'])
            if var_dict[field]['method'] == 'mean':
                superobs[field] = self.reduction_mean(qc_df, field, **var_dict[field]['reduction_kw'])
            else:
                print(f'reduction method {reduction} is not a valid option')
        
        return superobs


    def qc_obs(self, field='TQM', thres=2):
        """
        Quality control observations
        """

        qc_df = self.df.copy()

        # Apply quality marks
        qc_df = qc_df.loc[qc_df[field] <= thres]

        return qc_df

 
    def reduction_mean(self, qc_df, field):
        """
        Superob reduction using a regular mean
        """

        superob_groups = np.unique(self.df['superob_groups'])
        superobs = np.zeros(len(superob_groups)) * np.nan
        for i, g in enumerate(superob_groups):
            values = qc_df.loc[qc_df['superob_groups'] == g, field].values
            if len(values) > 0:
                superobs[i] = np.nanmean(values)

        return superobs
        

"""
End superob_prepbufr.py
"""

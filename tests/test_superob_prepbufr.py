"""
Tests for superob_prepbufr.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pytest
import pandas as pd
import xarray as xr

import pyDA_utils.superob_prepbufr as sp


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

class TestSuperob():

    @pytest.fixture(scope='class')
    def sample_pb(self):
        fname = './data/202204291500.rap.fake.prepbufr.for_superob_test.csv'
        return sp.superobPB(fname)
   

    def test_assign_superob(self, sample_pb):
        """
        Check that wrapper is passing arguments as expected
        """

        window = 120

        # Perform temporal grouping using the wrapper
        sample_pb.assign_superob('temporal', grouping_kw={'window':window})
        wrapper_df = sample_pb.df.copy()
        sample_pb.df['superob_groups'] = 0

        # Perform temporal grouping uing the grouping method directly
        sample_pb.grouping_temporal(window=120)

        assert np.all(wrapper_df['superob_groups'] ==  sample_pb.df['superob_groups'])


    def test_grouping_temporal(self, sample_pb):
        """
        Check if superob groups are created properly
        """

        # Create superob groups
        window=120
        window_dhr = window / 3600
        sample_pb.grouping_temporal(window=window)

        # Check that obs within a superob group are within window_dhr of one another
        cond = sample_pb.df['SID'] == np.unique(sample_pb.df['SID'].values)[0]
        dhr = sample_pb.df.loc[cond, 'DHR'].values
        groups = sample_pb.df.loc[cond, 'superob_groups'].values
        unique_groups = np.unique(groups)
        minvals = np.zeros(len(unique_groups))
        for i, g in enumerate(unique_groups):
            minvals[i] = np.amin(dhr[g == groups])
            assert (np.amax(dhr[g == groups]) - minvals[i]) <= window_dhr

        # Check that superob groups are at least window_dhr apart from one another
        minvals_sort = np.sort(minvals)
        print(minvals[1:] - minvals[:-1])
        assert np.amin(minvals[1:] - minvals[:-1]) >= (window_dhr - 1./3600.)

        # Check that superob groups from one SID do not overlap with another SID
        groups2 = sample_pb.df.loc[sample_pb.df['SID'] == np.unique(sample_pb.df['SID'].values)[1], 'superob_groups'].values
        for g in unique_groups:
            assert g not in groups2


    def test_grouping_grid(self, sample_pb):
        """
        For now, we'll just run this method to see if it crashes
        """

        sample_pb.grouping_grid(grid_fname='./data/RRFS_grid_max.nc')


    def test_map_proj_obs(self, sample_pb):
        """
        Check whether (x, y) from map projection lie within the RRFS grid domain
        """

        sample_pb.map_proj_obs()
        xcheck = np.logical_and(sample_pb.df['XMP'].values >= 0, 
                                sample_pb.df['XMP'].values <= 1799)
        ycheck = np.logical_and(sample_pb.df['YMP'].values >= 0, 
                                sample_pb.df['YMP'].values <= 1059)
              
        assert np.array_equal(xcheck, np.array([True]*len(sample_pb.df)))
        assert np.array_equal(ycheck, np.array([True]*len(sample_pb.df)))


    def test_interp_gridded_field_obs(self, sample_pb):
        """
        Check whether there is good agreement between the interpolated surface elevations and 
        the nearest neighbor
        """

        # Open file with RRFS grid information
        grid_ds = xr.open_dataset('./data/RRFS_grid_max.nc')
        hgt_sfc = grid_ds['HGT_SFC'].values
        x1d_grid = np.arange(grid_ds['lon'].shape[1])
        y1d_grid = np.arange(grid_ds['lon'].shape[0])

        sample_pb.interp_gridded_field_obs('SFC', (y1d_grid, x1d_grid), hgt_sfc)

        for s in np.unique(sample_pb.df['SID']):
            cond = sample_pb.df['SID'] == s
            i = int(np.around(sample_pb.df.loc[cond, 'YMP'].values[0]))
            j = int(np.around(sample_pb.df.loc[cond, 'XMP'].values[0]))
            assert np.isclose(sample_pb.df.loc[cond, 'SFC'].values[0], grid_ds['HGT_SFC'][i, j].values, atol=2)


    def test_reduction_superob(self, sample_pb):
        """
        Check that wrapper is performing as expected
        """

        # Run wrapper
        sample_pb.assign_superob('temporal')
        superobs = sample_pb.reduction_superob(var_dict={'TOB':{'method':'mean', 'qm_kw':{'field':'TQM', 'thres':2}, 'reduction_kw':{}},
                                                         'QOB':{'method':'mean', 'qm_kw':{'field':'QQM', 'thres':2}, 'reduction_kw':{}},
                                                         'POB':{'method':'mean', 'qm_kw':{'field':'PQM', 'thres':2}, 'reduction_kw':{}},
                                                         'YOB':{'method':'mean', 'qm_kw':{'field':'TQM', 'thres':2}, 'reduction_kw':{}},
                                                         'XOB':{'method':'mean', 'qm_kw':{'field':'TQM', 'thres':2}, 'reduction_kw':{}},
                                                         'DHR':{'method':'mean', 'qm_kw':{'field':'TQM', 'thres':2}, 'reduction_kw':{}}})

        # Check DHR manually
        qc_df = sample_pb.qc_obs(field='TQM', thres=2)
        superobs_DHR = sample_pb.reduction_mean(qc_df, 'DHR')
        
        assert np.all(superobs_DHR == superobs['DHR'].values)
        assert np.array_equal(superobs.columns, sample_pb.df.columns)

    
    def test_qc_obs(self, sample_pb):
        """
        Check QC procedure correctly removes rows with QM > 2
        """

        # Add some random instances with QM = 5
        ind = np.unique(np.random.randint(0, len(sample_pb.df), 50))
        sample_pb.df.loc[ind, 'TQM'] = 5

        # Perform QC
        qc_df = sample_pb.qc_obs(field='TQM', thres=2)
        assert np.all(qc_df['TQM'] <= 2)


    def test_reduction_mean(self, sample_pb):
        """
        Check that the average is applied correctly for a single superob
        """

        # Create input df
        sample_pb.assign_superob('temporal')
        qc_df = sample_pb.qc_obs(field='TQM', thres=2)
        group1 = qc_df['superob_groups'].values[0]

        # Create superobs
        superobs = sample_pb.reduction_mean(qc_df, 'TOB')

        assert superobs[0] == np.mean(qc_df.loc[qc_df['superob_groups'] == group1, 'TOB'])


"""
End test_superob_prepbufr.py
"""

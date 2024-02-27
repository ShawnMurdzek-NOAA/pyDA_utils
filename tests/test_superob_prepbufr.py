"""
Tests for superob_prepbufr.py

Note: Use -s option when running pytest to display stdout for tests that pass. Stdout should already
be displayed for tests that fail.

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pytest
import pandas as pd
import xarray as xr

import pyDA_utils.map_proj as mp
import pyDA_utils.superob_prepbufr as sp


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

class TestSuperob():

    @pytest.fixture(scope='class')
    def sample_pb(self):
        fname = './data/202204291500.rap.fake.prepbufr.for_superob_test.csv'
        sp_obj = sp.superobPB(fname)
        sp_obj.map_proj = mp.ll_to_xy_lc
        sp_obj.map_proj_kw = {'dx':3, 'knowni':899, 'knownj':529}
        return sp_obj
   

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
        assert np.amin(minvals[1:] - minvals[:-1]) >= (window_dhr - 1./3600.)

        # Check that superob groups from one SID do not overlap with another SID
        groups2 = sample_pb.df.loc[sample_pb.df['SID'] == np.unique(sample_pb.df['SID'].values)[1], 'superob_groups'].values
        for g in unique_groups:
            assert g not in groups2


    def test_grouping_grid(self, sample_pb):
        """
        Check that superob groups are created correctly
        """

        # Create superobs
        grid_fname='./data/RRFS_grid_max.nc'
        sample_pb.map_proj_kw = {'dx':3, 'knowni':899, 'knownj':529}
        sample_pb.grouping_grid(grid_fname=grid_fname)

        # Read in RRFS grid
        grid_ds = xr.open_dataset(grid_fname)
        zgrid = grid_ds['HGT_AGL'].values
        nz = len(zgrid)

        # Determine obs height AGL
        sample_pb.df['HGT_AGL'] = sample_pb.df['ZOB'].values - sample_pb.df['SFC'].values

        # Useful information when trying to debug why horizontal superob group is failing
        print('UA000001 XMP =', sample_pb.df.loc[sample_pb.df['SID'] == "'UA000001'", 'XMP'].values[0])
        print('UA000001 YMP =', sample_pb.df.loc[sample_pb.df['SID'] == "'UA000001'", 'YMP'].values[0])
        print('UA000002 XMP =', sample_pb.df.loc[sample_pb.df['SID'] == "'UA000002'", 'XMP'].values[0])
        print('UA000002 YMP =', sample_pb.df.loc[sample_pb.df['SID'] == "'UA000002'", 'YMP'].values[0])

        # Examine superob groups
        all_sid = np.unique(sample_pb.df['SID'].values)
        ngroupsv = 0
        ngroupsh = 0
        for i in range(nz-1, 1, -1):

            # Examine superob groups in the vertical (this test only works for vertical profiles)
            for sid in all_sid:
                target_group = np.unique(sample_pb.df.loc[(sample_pb.df['HGT_AGL'] >= zgrid[i]) &
                                                          (sample_pb.df['HGT_AGL'] < zgrid[i-1]) &
                                                          (sample_pb.df['SID'] == sid), 'superob_groups'])
                other_groups = np.unique(sample_pb.df.loc[(sample_pb.df['HGT_AGL'] < zgrid[i]) |
                                                          (sample_pb.df['HGT_AGL'] >= zgrid[i-1]) &
                                                          (sample_pb.df['SID'] == sid), 'superob_groups'])
                if len(target_group) > 0:
                    ngroupsv = ngroupsv + 1
                    assert len(target_group) == 1
                    assert target_group[0] not in other_groups

            # Examine superob groups in the horizontal
            # Use UA000001 and UA000002, which are designed to be in the same horizontal superob groups
            target_group = np.unique(sample_pb.df.loc[(sample_pb.df['HGT_AGL'] >= zgrid[i]) &
                                                      (sample_pb.df['HGT_AGL'] < zgrid[i-1]) &
                                                      ((sample_pb.df['SID'] == "'UA000001'") |
                                                       (sample_pb.df['SID'] == "'UA000002'")), 'superob_groups'])
            other_groups = np.unique(sample_pb.df.loc[(sample_pb.df['SID'] != "'UA000001'") &
                                                      (sample_pb.df['SID'] != "'UA000002'"), 'superob_groups'])
                
            if len(target_group) > 0:
                ngroupsh = ngroupsh + 1
                assert len(target_group) == 1
                assert target_group[0] not in other_groups

        # This ensures that we captured at least one superob group in the vertical and horizontal
        print('ngroups in vertical =', ngroupsv)
        print('ngroups in horizontal =', ngroupsh)
        assert ngroupsv > 1
        assert ngroupsh > 1


    def test_map_proj_obs(self, sample_pb):
        """
        Check whether (x, y) from map projection lie within the RRFS grid domain
        """

        sample_pb.df = sample_pb.map_proj_obs(sample_pb.df)
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


    def test_reduction_hor_cressman(self, sample_pb):
        """
        We'll just try running this method for now
        """

        # Obtain superob coordinates
        superobs_in = sample_pb.reduction_superob(var_dict={})

        # Create input df
        sample_pb.assign_superob('temporal')
        qc_df = sample_pb.qc_obs(field='TQM', thres=2)
        group1 = qc_df['superob_groups'].values[0]

        sample_pb.reduction_hor_cressman(qc_df, superobs_in, 'TOB')


"""
End test_superob_prepbufr.py
"""

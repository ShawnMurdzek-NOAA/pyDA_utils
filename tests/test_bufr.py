"""
Tests for bufr.py

Note: Use -s option when running pytest to display stdout for tests that pass. Stdout should already
be displayed for tests that fail.

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pytest
import copy
import random
from haversine import haversine_vector

import pyDA_utils.bufr as bufr


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

class TestBUFR():

    @pytest.fixture(scope='class')
    def sample_pb(self):
        fname = './data/202202011200.rap.prepbufr.for_bufr_test.csv'
        return bufr.bufrCSV(fname)
   
    
    @pytest.fixture(scope='class')
    def sample_add_err_pb(self):
        """
        Sample data for testing code to add observation errors
        """

        # Generate data
        nobs = 500
        nsid = 2
        DHR = np.array(nsid*list(range(nobs))) / 3600.
        SID = []
        for i in range(nsid):
            SID = SID + nobs*[f'SID{i}']
        TOB = 50*DHR + 270
        POB = 1000 - 500*DHR

        # Shuffle data
        idx = list(range(len(DHR)))
        random.shuffle(idx)
        test_dict = {'SID':np.array(SID)[idx], 'DHR':DHR[idx], 'TOB':TOB[idx], 'POB':POB[idx]}

        return pd.DataFrame(test_dict)


    def test_select_obtypes(self, sample_pb):
        """
        Test method to select ob types
        """

        tmp_bufr = copy.deepcopy(sample_pb)
        tmp_bufr.select_obtypes([187])
        assert np.unique(tmp_bufr.df['TYP']) == np.array([187])
    

    def test_select_dhr(self, sample_pb):
        """
        Test method to select ob valid time
        """

        tmp_bufr = copy.deepcopy(sample_pb)
        tmp_bufr.select_dhr(0)

        # Specific example ("'SKBO'", which should have a DHR of 0 after applying select_dhr)
        assert np.all(np.isclose(tmp_bufr.df['DHR'].loc[tmp_bufr.df['SID'] == "'SKBO'"], 0))

        # Check that SKBO has two entries (both for type 187, CLAM = 13 and 11 at HOCB = 520 and 6100)
        assert len(tmp_bufr.df.loc[tmp_bufr.df['SID'] == "'SKBO'"]) == 2

        # Check to ensure that no TYP/SID combo has > 1 unique DHR after applying select_dhr
        # Also check that some TYP/SID combos have > 1 entries after applying select_dhr (this is expected if there is cloud info)
        n_dhr = []
        n_entries = []
        for t in np.unique(tmp_bufr.df['TYP']):
            t_cond = tmp_bufr.df['TYP'] == t
            for s in np.unique(tmp_bufr.df['SID'].loc[t_cond]):
                subset_df = tmp_bufr.df.loc[t_cond & (tmp_bufr.df['SID'] == s)]
                n_dhr.append(len(np.unique(subset_df['DHR'])))
                n_entries.append(len(subset_df))
        assert np.all(np.array(n_dhr) == 1)
        assert not np.all(np.array(n_entries) == 1)


    def test_select_latlon(self, sample_pb):
        """
        Test method to only select observations in a (lat, lon) box
        """

        tmp_bufr = copy.deepcopy(sample_pb)

        # Size of DataFrame before restrictions
        init_len = len(tmp_bufr.df)

        # Restrict obs to be in a smaller box
        tmp_bufr.select_latlon(35, 250, 45, 270)

        # Check that DataFrame is smaller after restricting obs to a smaller domain
        assert len(tmp_bufr.df) < init_len
        
        # Check that none of the (lat, lon) coordinates fall outside the desired box
        assert tmp_bufr.df['XOB'].max() <= 270
        assert tmp_bufr.df['XOB'].min() >= 250
        assert tmp_bufr.df['YOB'].max() <= 45
        assert tmp_bufr.df['YOB'].min() >= 35


    def test_match_types(self, sample_pb):
        """
        Test method to match thermodynamic and kinematic obs together
        """

        tmp_bufr = copy.deepcopy(sample_pb)

        # Match two aircraft fields
        tmp_bufr.match_types(233, 133, match_fields=['SID', 'XOB', 'YOB'], copy_fields=['TOB'])
        assert np.all(tmp_bufr.df.loc[tmp_bufr.df['TYP'] == 187, 'match'] == 0)
        print()
        print("In test_match_types")
        for m in range(1, np.amax(tmp_bufr.df['match'])+1):
            print(m)
            cond = tmp_bufr.df['match'] == m
            for f in ['SID', 'XOB', 'YOB', 'DHR', 'TOB']:
                assert len(np.unique(tmp_bufr.df.loc[cond, f])) == 1

        # Match GPS IPW to an undefined field
        tmp_bufr.match_types(153, 200, match_fields=['SID', 'XOB', 'YOB'])
        assert np.all(tmp_bufr.df.loc[tmp_bufr.df['TYP'] == 153, 'match'] == -1)


    def test_compute_ceil(self, sample_pb):
        """
        Test ceiling computations
        """
  
        # Only retain ob type 187
        subset_df = sample_pb.df.loc[np.isclose(sample_pb.df['TYP'], 187), :]

        # Compute ceilings and extract necessary fields
        ceil_compute = bufr.compute_ceil(subset_df, use_typ=[187], no_ceil=2e4)
        ceil_raw = subset_df.loc[:, 'CEILING'].values
        clam = subset_df.loc[:, 'CLAM'].values
        hocb = subset_df.loc[:, 'HOCB'].values

        # Set NaN to 1e9 (b/c NaN == NaN is False)
        ceil_compute[np.isnan(ceil_compute)] = 1e9
        ceil_raw[np.isnan(ceil_raw)] = 1e9

        # CEILING encodes CLAM = 9 as "no ceiling" if HOCB = NaN. I don't agree with this, so
        # CEILING2 encodes this as "missing". Check that this is indeed the case
        idx = np.where(np.logical_and(np.isnan(hocb), np.isclose(clam, 9)))[0]
        assert np.all(np.isclose(ceil_compute[idx], 1e9))
        ceil_raw[idx] = 1e9

        # Check that ceilings match
        assert np.all(np.isclose(ceil_compute, ceil_raw))


    def test_create_uncorr_obs_err(self):
        """
        Test method to create uncorrelated observation errors
        """

        # Constant stdev
        err = bufr.create_uncorr_obs_err(100000, 0.5)
        assert np.isclose(np.std(err), 0.5, atol=0.02)
        assert np.isclose(np.mean(err), 0, atol=0.02)

        # Variable stdev
        stdev = np.array(1000*[0.5] + 1000*[5])
        err = bufr.create_uncorr_obs_err(len(stdev), stdev)
        assert np.isclose(np.std(err[:1000]), 0.5, atol=0.05)
        assert np.isclose(np.std(err[1000:]), 5, atol=0.5)
        assert np.isclose(np.mean(err), 0, atol=0.3)


    def test_create_corr_obs_err(self, sample_add_err_pb):
        """
        Test method to create correlated observation errors
        """

        # Constant stdev
        err = bufr.create_corr_obs_err(sample_add_err_pb, 1, 'DHR', auto_reg_parm=0.5, min_d=1)
        err_ID0 = err[sample_add_err_pb['SID'].values == 'SID0']
        err_ID0_sorted = err_ID0[np.argsort(sample_add_err_pb.loc[sample_add_err_pb['SID'] == 'SID0', 'DHR'].values)]
        autocorr = np.corrcoef(err_ID0_sorted[1:], err_ID0_sorted[:-1])[0, 1]
        print()
        print("In test_create_corr_obs_err")
        print(f"constant stdev autocorr = {autocorr} (should be close to 0.5)")
        assert np.isclose(autocorr, 0.5, atol=0.1)

        # Constant stdev special case: autocorrelation should be close to 0 b/c min_d is small
        err = bufr.create_corr_obs_err(sample_add_err_pb, 1, 'DHR', auto_reg_parm=0.5, min_d=1e-10)
        err_ID0 = err[sample_add_err_pb['SID'].values == 'SID0']
        err_ID0_sorted = err_ID0[np.argsort(sample_add_err_pb.loc[sample_add_err_pb['SID'] == 'SID0', 'DHR'].values)]
        autocorr = np.corrcoef(err_ID0_sorted[1:], err_ID0_sorted[:-1])[0, 1]
        print(f"constant stdev special case autocorr = {autocorr} (should be close to 0)")
        assert np.isclose(autocorr, 0, atol=0.1)
        assert np.isclose(np.std(err_ID0_sorted), 1, atol=0.1)

        # Constant stdev, but with partition_dim feature using POB
        # Change POB to be constant for the first half of the dataset so that the autocorrelation is nonzero
        # B/c POB is different for every ob, the autocorrelation should be close to 0
        tmp_df = copy.deepcopy(sample_add_err_pb)
        tmp_df.loc[tmp_df['POB'] > 965, 'POB'] = 1000
        err = bufr.create_corr_obs_err(tmp_df, 1, 'DHR', partition_dim='POB', auto_reg_parm=0.5, min_d=1)
        err_ID0_1 = err[np.logical_and(tmp_df['SID'].values == 'SID0', tmp_df['POB'] > 965)]
        err_ID0_2 = err[np.logical_and(tmp_df['SID'].values == 'SID0', tmp_df['POB'] <= 965)]
        err_ID0_1_sorted = err_ID0_1[np.argsort(tmp_df.loc[np.logical_and(tmp_df['SID'].values == 'SID0', tmp_df['POB'] > 965), 'DHR'].values)]
        err_ID0_2_sorted = err_ID0_2[np.argsort(tmp_df.loc[np.logical_and(tmp_df['SID'].values == 'SID0', tmp_df['POB'] <= 965), 'DHR'].values)]
        autocorr1 = np.corrcoef(err_ID0_1_sorted[1:], err_ID0_1_sorted[:-1])[0, 1]
        autocorr2 = np.corrcoef(err_ID0_2_sorted[1:], err_ID0_2_sorted[:-1])[0, 1]
        print(f"constant stdev with POB partition_dim autocorr (first half) = {autocorr1} (should be close to 0.5)")
        assert np.isclose(autocorr1, 0.5, atol=0.15)
        print(f"constant stdev with POB partition_dim autocorr (second half) = {autocorr2} (should be close to 0)")
        assert np.isclose(autocorr2, 0, atol=0.15)

        # Variable stdev
        df_ID0 = sample_add_err_pb.loc[sample_add_err_pb['SID'] == 'SID0']
        stdev = np.array(int(len(df_ID0)/2)*[0.5] + int(len(df_ID0)/2)*[5])
        err = bufr.create_corr_obs_err(df_ID0, stdev, 'DHR', auto_reg_parm=0.5, min_d=1)
        err_sorted = err[np.argsort(df_ID0['DHR'].values)]
        autocorr = np.corrcoef(err_sorted[1:], err_sorted[:-1])[0, 1]
        print(f"variable stdev autocorr = {autocorr} (should be close to 0.5)")
        assert np.isclose(autocorr, 0.5, atol=0.1)

        # Variable stdev special case: autocorrelation should be close to 0 b/c min_d is small
        df_ID0 = sample_add_err_pb.loc[sample_add_err_pb['SID'] == 'SID0']
        nhalf = int(len(df_ID0)/2)
        stdev = np.array(nhalf*[0.5] + nhalf*[5])
        err = bufr.create_corr_obs_err(df_ID0, stdev, 'DHR', auto_reg_parm=0.5, min_d=1e-10)
        err_sorted = err[np.argsort(df_ID0['DHR'].values)]
        autocorr = np.corrcoef(err_sorted[1:], err_sorted[:-1])[0, 1]
        print(f"variable stdev special case autocorr = {autocorr} (should be close to 0)")
        assert np.isclose(autocorr, 0, atol=0.1)
        assert np.isclose(np.std(err[:nhalf]), 0.5, atol=0.05)
        assert np.isclose(np.std(err[nhalf:]), 5, atol=0.5)
        assert np.isclose(np.mean(err), 0, atol=0.3)


    def test_thin_obs(self, sample_pb):
        """
        Test function to thin obs
        """

        tmp_bufr = copy.deepcopy(sample_pb)
        radius = 50000.  # meters

        # Check that there are some points within the thinning radius of the first ob
        all_pts = np.array([tmp_bufr.df['YOB'].values, tmp_bufr.df['XOB'] - 360]).T
        pt1 = np.array([tmp_bufr.df['YOB'].values[0], tmp_bufr.df['XOB'].values[0] - 360])
        dist = np.squeeze(haversine_vector(all_pts, pt1, check=False, comb=True))
        assert np.amin(dist[1:]) < (1e-3 * radius)

        # Apply thinning, then check that there are no points in the thinning radius
        thin_df = bufr.thin_obs(tmp_bufr.df, radius=radius)
        all_pts = np.array([thin_df['YOB'].values, thin_df['XOB'] - 360]).T
        pt1 = np.array([thin_df['YOB'].values[0], thin_df['XOB'].values[0] - 360])
        dist = np.squeeze(haversine_vector(all_pts, pt1, check=False, comb=True))
        assert len(thin_df) < len(tmp_bufr.df)
        assert np.amin(dist[1:]) >= (1e-3 * radius)


"""
End test_bufr.py
"""

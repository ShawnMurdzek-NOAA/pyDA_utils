"""
Tests for bufr.py

Note: Use -s option when running pytest to display stdout for tests that pass. Stdout should already
be displayed for tests that fail.

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pytest
import copy

import pyDA_utils.bufr as bufr


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

class TestBUFR():

    @pytest.fixture(scope='class')
    def sample_pb(self):
        fname = './data/202202011200.rap.prepbufr.for_bufr_test.csv'
        return bufr.bufrCSV(fname)
   

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


    def test_match_types(self, sample_pb):
        """
        Test method to match thermodynamic and kinematic obs together
        """

        tmp_bufr = copy.deepcopy(sample_pb)

        # Match two aircraft fields
        tmp_bufr.match_types(233, 133, match_fields=['SID', 'XOB', 'YOB'], copy_fields=['TOB'])
        assert np.all(tmp_bufr.df.loc[tmp_bufr.df['TYP'] == 187, 'match'] == 0)
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


"""
End test_bufr.py
"""

"""
Tests for limit_prpebufr.py

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
import pyDA_utils.limit_prepbufr as lp


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

class TestBUFRlim():

    @pytest.fixture(scope='class')
    def sample_pb(self):
        fname = './data/202204291500.rap.fake.prepbufr.for_superob_test.csv'
        return bufr.bufrCSV(fname)
   

    def test_wspd_limit(self, sample_pb):

        tmp_bufr = copy.deepcopy(sample_pb)
        tmp_bufr = lp.wspd_limit(tmp_bufr, lim=15)

        # Check that all rows with WSPD > 15 are flagged
        assert np.all((tmp_bufr.df.loc[tmp_bufr.df['TYP'] == 236, 'WSPD'] > 15) == 
                      tmp_bufr.df.loc[tmp_bufr.df['TYP'] == 236, 'cond'])


    def test_detect_icing(self, sample_pb):

        tmp_bufr = copy.deepcopy(sample_pb)
        tmp_bufr = lp.detect_icing(tmp_bufr, tob_lim=2, rh_lim=90)

        # Check that all rows meeting icing conditions are flagged
        assert np.all(np.logical_and((tmp_bufr.df.loc[tmp_bufr.df['TYP'] == 136, 'TOB'] < 2),
                                     (tmp_bufr.df.loc[tmp_bufr.df['TYP'] == 136, 'RHOB'] > 90)) ==
                      tmp_bufr.df.loc[tmp_bufr.df['TYP'] == 136, 'cond'])


    def test_remove_obs_after_lim(self, sample_pb):

        tmp_bufr = copy.deepcopy(sample_pb)

        # Change the WSPD and cond fields for UA000001
        idx = list(range(848, 853)) + list(range(870, 890))
        tmp_bufr.df.loc[idx, 'VOB'] = 5
        tmp_bufr.df.loc[idx, 'cond'] = False
        last_dhr_uas1 = tmp_bufr.df.loc[854, 'DHR']

        tmp_bufr = lp.wspd_limit(tmp_bufr, lim=15)
        df = lp.remove_obs_after_lim(tmp_bufr.df, 236, match_type=[136], nthres=3)

        # Check that the max DHR for UA000001 is last_dhr_uas1
        assert np.amax(df.loc[df['SID'] == "'UA000001'", 'DHR']) == last_dhr_uas1

        # Check that only 4 WSPDs exceed 15 m/s for UA000001
        assert np.sum(df.loc[df['SID'] == "'UA000001'", 'WSPD'] > 15) == 4

        # Check that only 2 WSPDs exceed 15 m/s for UA000002
        assert np.sum(df.loc[df['SID'] == "'UA000002'", 'WSPD'] > 15) == 2

        # Check that there are the same number of 236 and 136 obs
        assert len(df.loc[df['TYP'] == 136]) == len(df.loc[df['TYP'] == 236])


"""
End test_limit_prepbufr.py
"""

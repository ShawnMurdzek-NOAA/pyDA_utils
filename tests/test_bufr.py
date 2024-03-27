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
import pandas as pd
import xarray as xr
import metpy.interpolate as mi

import pyDA_utils.bufr as bufr
import pyDA_utils.meteo_util as mu


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

class TestBUFR():

    @pytest.fixture(scope='class')
    def sample_pb(self):
        fname = './data/202202011200.rap.prepbufr.for_bufr_test.csv'
        return bufr.bufrCSV(fname)
   

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

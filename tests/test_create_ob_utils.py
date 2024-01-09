"""
Tests for create_ob_utils.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pytest

import pyDA_utils.create_ob_utils as cou


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

def test_check_wind_ref_frame():

    # Should use some smaller GRIB2 files that I can keep on GitHub for this, but this quick fix
    # works for now
    fname_grid_rel = '/work2/noaa/wrfruc/murdzek/nature_run_winter/UPP/20220201/wrfnat_202202010000.grib2'
    fname_earth_rel = '/work2/noaa/wrfruc/murdzek/nature_run_winter/UPP/20220201/wrfnat_202202010000_er.grib2'

    assert not cou.check_wind_ref_frame(fname_grid_rel)
    assert cou.check_wind_ref_frame(fname_earth_rel)


"""
End test_create_ob_utils.py
"""

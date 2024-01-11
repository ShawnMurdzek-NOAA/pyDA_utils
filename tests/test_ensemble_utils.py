"""
Tests for ensemble_utils.py

Environment: adb_graphics (Jet)
shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pytest
import yaml

import pyDA_utils.ensemble_utils as eu


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

class TestEnsemble():

    @pytest.fixture(scope='class')
    def sample_ens(self):
    
        # Load YAML with parameters for ensemble class
        fname = 'test_ensemble_utils.yml'
        with open(fname, 'r') as fptr:
            param = yaml.safe_load(fptr)

        # Format UPP file names correctly
        str_format = param['str_format']
        prslev_fnames = {}
        natlev_fnames = {}
        for i in range(1, param['nmem']+1):
            prslev_fnames['mem{num:04d}'.format(num=i)] = str_format.format(num=i, lev='prslev')
            natlev_fnames['mem{num:04d}'.format(num=i)] = str_format.format(num=i, lev='natlev')
        
        return eu.ensemble(natlev_fnames, extra_fnames=prslev_fnames, 
                           extra_fields=param['prslev_vars'], 
                           bufr_csv_fname=param['bufr_fname'], 
                           lat_limits=[param['min_lat'], param['max_lat']],
                           lon_limits=[param['min_lon'], param['max_lon']],
                           zind=param['z_ind'],
                           state_fields=param['state_vars'],
                           bec=True)


    def test_subset_bufr(self, sample_ens):
        nonan_field = 'TOB'
        subset_bufr = sample_ens._subset_bufr(['ADPSFC'], nonan_field, DHR=0)

        # Check spatial subsetting
        assert np.amax(subset_bufr['XOB'] - 360.) <= sample_ens.lon_limits[1]
        assert np.amin(subset_bufr['XOB'] - 360.) >= sample_ens.lon_limits[0]
        assert np.amax(subset_bufr['YOB']) <= sample_ens.lat_limits[1]
        assert np.amin(subset_bufr['YOB']) >= sample_ens.lat_limits[0]
  
        # Check that nonan field was implemented properly
        assert np.all(~np.isnan(subset_bufr[nonan_field]))

        # Check that only a single ob is retained for each SID
        assert np.array_equal(np.unique(subset_bufr['SID']), subset_bufr['SID'])

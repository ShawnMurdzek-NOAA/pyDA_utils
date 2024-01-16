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
import pandas as pd

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


    def test_interp_model_2d(self, sample_ens):
        mem = sample_ens.mem_names[0]
        model_lat = np.ravel(sample_ens.subset_ds[mem]['gridlat_0'][2:-2:5, 2:-2:5].values)
        model_lon = np.ravel(sample_ens.subset_ds[mem]['gridlon_0'][2:-2:5, 2:-2:5].values)
        ob_lat = model_lat + np.random.uniform(-0.002, 0.002, size=model_lat.size)
        ob_lon = model_lon + np.random.uniform(-0.002, 0.002, size=model_lon.size)

        # Test nearest neighbor interpolation
        # Use model lat coordinates with small perturbations as test data
        # Assuming 1 km = 0.01 deg, the distance between two adjacent gridpoints should be ~0.03 deg
        # B/c ob locations differ from model gridpoints by no more than 0.002*sqrt(2) deg, model_lat
        # should match the interpolated lats
        near_neigh_lat_df = sample_ens.interp_model_2d('gridlat_0', ob_lat, ob_lon, method='nearest')
        assert np.allclose(near_neigh_lat_df[mem].values, model_lat)

        # Test linear interpolation
        # Create some random linear data
        def create_linear_data(lat, lon):
            return 2*lat + 3*lon + 4
        for m in sample_ens.mem_names:
            sample_ens.subset_ds[m]['interp_test'] = create_linear_data(sample_ens.subset_ds[m]['gridlat_0'],
                                                                        sample_ens.subset_ds[m]['gridlon_0'])
        linear_df = sample_ens.interp_model_2d('interp_test', ob_lat, ob_lon, method='linear')
        assert np.allclose(linear_df[mem].values, create_linear_data(ob_lat, ob_lon))



"""
End test_ensemble_utils.py
"""

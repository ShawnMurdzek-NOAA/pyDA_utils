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
import matplotlib.pyplot as plt

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


    def test_create_state_matrix(self, sample_ens):
        # Note: This test only examines the case where thin = 1

        # First, check that the length of the state matrix is appropriate
        sample_ds = sample_ens.subset_ds[sample_ens.mem_names[0]]
        state_matrix = sample_ens.state_matrix
        state_vars = np.unique(state_matrix['vars'])
        for v in state_vars:
            assert len(state_matrix['data'][state_matrix['vars'] == v, 0]) == sample_ds[v].size
        
        """
        # Next, perform some spot checks for the state_matrix locations
        # Most comparisons have len(idx) > 1, but that is fine as long as the first match is from (i, j, k)
        # Cloud cover (TCDC) is ignored b/c there are several matches before (i, j, k)
        for v in ['SPFH_P0_L105_GLC0', 'TMP_P0_L105_GLC0']:
            sub_state = state_matrix['data'][state_matrix['vars'] == v, 0]
            sub_loc = state_matrix['loc'][state_matrix['vars'] == v]
            i = 0
            for j in [0, 1]:
                for k in [1, 4]:
                    val = sample_ds[v].values[i, j, k]
                    idx = np.where(np.isclose(sub_state, val))[0]
                    print(f"v={v}, j={j}, k={k}, idx=", idx)
                    idx = idx[0]
                    assert np.isclose(sample_ds['lv_HYBL2'][i], sub_loc[idx, 0])
                    assert np.isclose(sample_ds['gridlat_0'][j, k], sub_loc[idx, 1])
                    assert np.isclose(sample_ds['gridlon_0'][j, k], sub_loc[idx, 2])
        """
        
        # See if we can recover the proper 3D arrays from the state matrix
        shape3d = sample_ds[state_vars[0]].shape
        for v in state_vars:
            var3d = np.reshape(state_matrix['data'][state_matrix['vars'] == v, 0], shape3d)
            z3d = np.reshape(state_matrix['loc'][state_matrix['vars'] == v, 0], shape3d)
            lat3d = np.reshape(state_matrix['loc'][state_matrix['vars'] == v, 1], shape3d)
            lon3d = np.reshape(state_matrix['loc'][state_matrix['vars'] == v, 2], shape3d)

            assert np.allclose(sample_ds[v].values, var3d)
            assert z3d[0, 0, 0] == sample_ds['lv_HYBL2'][0]
            assert np.allclose(sample_ds['gridlat_0'].values, lat3d[0, :, :])
            assert np.allclose(sample_ds['gridlon_0'].values, lon3d[0, :, :])


    def test_check_pts_in_subset_domain(self, sample_ens):
        ctr_lat = 0.5 * (sample_ens.lat_limits[0] + sample_ens.lat_limits[1])
        ctr_lon = 0.5 * (sample_ens.lon_limits[0] + sample_ens.lon_limits[1])
        points = [[ctr_lon, ctr_lat],
                  [sample_ens.lon_limits[0] - 0.1, ctr_lat],
                  [sample_ens.lon_limits[1] + 0.1, ctr_lat],
                  [ctr_lon, sample_ens.lat_limits[0] - 0.1],
                  [ctr_lon, sample_ens.lat_limits[1] + 0.1]]
        indomain = sample_ens.check_pts_in_subset_domain(points)
              
        assert np.array_equal(indomain, np.array([True, False, False, False, False]))


    def test_subset_bufr(self, sample_ens):
        nonan_field = 'TOB'
        subset_bufr = sample_ens._subset_bufr(['ADPSFC'], nonan_field, DHR=0)

        # Check that at least some obs were retained
        assert len(subset_bufr) > 0

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


    def test_skewt(self, sample_ens):
        # This test just checks whether the code runs without errors
        lat = 0.5*(sample_ens.lat_limits[0] + sample_ens.lat_limits[1])
        lon = 0.5*(sample_ens.lon_limits[0] + sample_ens.lon_limits[1])
        fig = plt.figure(figsize=(8, 8))
        skew = sample_ens.plot_skewts(lon, lat, fig, skew_kw={'hodo':False, 'barbs':False})


"""
End test_ensemble_utils.py
"""
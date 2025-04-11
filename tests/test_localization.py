"""
Tests for localization.py

Note: Use -s option when running pytest to display stdout for tests that pass. Stdout should already
be displayed for tests that fail.

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pytest

import pyDA_utils.localization as local


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

def test_gaspari_cohn_5ord():

    # Check ValueError is raised if dist is not an array
    with pytest.raises(ValueError):
        local.gaspari_cohn_5ord(1, 1)
    
    # Check ValueError is raised if dist is not >= 0
    with pytest.raises(ValueError):
        local.gaspari_cohn_5ord(np.arange(-1, 1, 4), 1)
    
    # Check end values of localization function and that localization function is in range [0, 1]
    C = local.gaspari_cohn_5ord(np.linspace(0, 4, 20), 1)
    assert np.isclose(C[0], 1)
    assert np.all(np.isclose(C[10:], 0))
    assert np.all(np.logical_and(C >= 0, C <= 1))


class TestLocalization():

    @pytest.fixture(scope='class')
    def sample_local(self):
        return local.localization_fct(local.gaspari_cohn_5ord)
    
    @pytest.fixture(scope='class')
    def sample_coords(self):
        # Truth distance is verified using https://www.nhc.noaa.gov/gccalc.shtml
        coords = {'model_pts':np.array([[39.06, -108.55],  # Grand Junction, CO
                                        [25.76, -80.19],   # Miami, FL
                                        [-33.86, 151.21]]), # Sydney, Australia
                  'ob_pt': np.array([40.02, -105.27]),  # Boulder, CO
                  'truth_dist': np.array([301.4109, 2813.8, 13385.8691])}
        return coords
    

    def test_compute_latlon_dist_geodesic(self, sample_local, sample_coords):
        dist = sample_local.compute_latlon_dist(sample_coords['model_pts'], sample_coords['ob_pt'], method='geodesic')
        assert np.allclose(dist, sample_coords['truth_dist'], rtol=1e-5)


    def test_compute_latlon_dist_haversine(self, sample_local, sample_coords):
        dist = sample_local.compute_latlon_dist(sample_coords['model_pts'], sample_coords['ob_pt'], method='haversine')
        assert np.allclose(dist, sample_coords['truth_dist'], rtol=0.005)


    def test_compute_partial_localization(self, sample_local, sample_coords):
        
        # Test localization with 1D coords
        model_coord = np.arange(10)
        ob_coord = 5
        C = sample_local.compute_partial_localization(model_coord, ob_coord, 2)
        assert np.isclose(C[5], 1)
        assert np.isclose(C[0], 0)
        assert np.isclose(C[9], 0)
        assert np.all(np.logical_and(C >= -1e-12, C <= 1+1e12))

        # Test localization with 2D coords
        C = sample_local.compute_partial_localization(sample_coords['model_pts'], sample_coords['ob_pt'], 500)
        assert C[0] > 0
        assert np.isclose(C[1], 0)
        assert np.all(np.logical_and(C >= -1e-12, C <= 1+1e12))
    

    def test_compute_localization(self, sample_local, sample_coords):
        
        # Assemble test coordinates
        ob_coord = np.array([1.5, sample_coords['ob_pt'][0], sample_coords['ob_pt'][1]])
        model_2d_pts = sample_coords['model_pts']
        n_2d_pts = model_2d_pts.shape[0]
        model_coords = np.zeros([n_2d_pts*5, 3])
        for i in np.arange(5):
            for j in range(n_2d_pts):
                model_coords[i*n_2d_pts+j, :] = np.array([i, model_2d_pts[j, 0], model_2d_pts[j, 1]])
        print('model_coords =', model_coords)
        
        # Compute localization
        C = sample_local.compute_localization(model_coords, ob_coord, 1, 500)
        print('C =', C)

        assert C[3] > 0
        assert C[6] > 0
        assert np.isclose(C[12], 0)
        assert np.all(np.logical_and(C >= -1e-12, C <= 1+1e12))


"""
End test_localization.py
"""

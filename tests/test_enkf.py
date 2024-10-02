"""
Tests for enkf.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pytest
import pandas as pd

from pyDA_utils import enkf


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

class TestEnKF():

    @pytest.fixture(scope='class')
    def sample_enkf(self):
    
        # Generate fake ensemble members using a sine function with added random noise
        m = 30
        N = 10
        x_b = np.zeros([m, N])
        loc = np.linspace(0, 4*np.pi, m)
        rng = np.random.default_rng(seed=10)
        for i in range(N):
            x_b[:, i] = 4 * np.sin(loc) + (rng.random(m) - 0.5) + 0.5*(rng.random(m) - 0.3)
        
        # Generate a fake observation at loc[10]
        y_0 = 4 * np.sin(loc[10]) + 0.25 * ((rng.random(1))[0] - 0.5)
        Hx_b = x_b[10, :]
        ob_var = (0.3)**2

        # The "truth"
        x_T = 4 * np.sin(loc)

        return enkf.enkf_1ob(x_b, y_0, Hx_b, ob_var), x_T


    def test_EnSRF(self, sample_enkf):
    
        enkf_obj = sample_enkf[0]
        x_T = sample_enkf[1]

        # Perform EnSRF
        enkf_obj.EnSRF()

        # Compare background and analysis mean to the truth
        rmse_b = np.sqrt(np.mean(((enkf_obj.x_b_bar - x_T)**2)))
        rmse_a = np.sqrt(np.mean(((enkf_obj.x_a_bar - x_T)**2)))
        print()
        print(f'Background Mean RMSE = {rmse_b}')
        print(f'Analysis Mean RMSE = {rmse_a}')
        assert rmse_a < rmse_b


"""
End test_enkf.py
"""
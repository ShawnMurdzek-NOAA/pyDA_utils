"""
Tests for Variational DA code

Initial code draft from Google Gemini

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from pyDA_utils.variational_da import ThreeDVar


#---------------------------------------------------------------------------------------------------
# Fixtures
#---------------------------------------------------------------------------------------------------

@pytest.fixture
def linear_setup():
    """Provides a simple linear setup where math can be easily verified."""
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    R = np.array([[2.0, 0.0], [0.0, 2.0]])
    
    def H(x):
        return x  # Identity observation operator

    return B, R, H


@pytest.fixture
def nonlinear_setup():
    """Provides the nonlinear setup from the original example."""
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    R = np.array([[0.05, 0.0], [0.0, 0.05]])
    
    def H(x):
        return np.array([x[0], x[1]**2])
        
    return B, R, H


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

def test_initialization(linear_setup):
    """Tests if the covariance matrices are inverted and cached correctly."""
    B, R, H = linear_setup
    da = ThreeDVar(B, R, H)
    
    expected_B_inv = np.array([[0.5, 0.0], [0.0, 0.5]])
    expected_R_inv = np.array([[0.5, 0.0], [0.0, 0.5]])
    
    assert_array_almost_equal(da.B_inv, expected_B_inv)
    assert_array_almost_equal(da.R_inv, expected_R_inv)


def test_cost_function_calculation(linear_setup):
    """Tests the cost function math using known manual calculations."""
    B, R, H = linear_setup
    da = ThreeDVar(B, R, H)
    
    xb = np.array([0.0, 0.0])
    y = np.array([2.0, 2.0])
    x_test = np.array([1.0, 1.0])
    
    # Manual Math:
    # J_b = 0.5 * (1-0)^2 * 0.5 + 0.5 * (1-0)^2 * 0.5 = 0.25 + 0.25 = 0.5
    # J_o = 0.5 * (2-1)^2 * 0.5 + 0.5 * (2-1)^2 * 0.5 = 0.25 + 0.25 = 0.5
    # Total J = 1.0
    cost = da._cost_function(x_test, xb, y)
    
    assert cost == 1.0


def test_assimilate_linear_exact_solution(linear_setup):
    """
    Tests if a simple linear assimilation finds the exact middle point.
    Since B and R are identical, the analysis should be exactly 
    halfway between the background and observation.
    """
    B, R, H = linear_setup
    da = ThreeDVar(B, R, H)
    
    xb = np.array([0.0, 0.0])
    y = np.array([2.0, 2.0])
    
    xa, res = da.assimilate(xb, y)
    
    assert res.success is True
    assert_allclose(xa, np.array([1.0, 1.0]), rtol=1e-5)


def test_assimilate_nonlinear(nonlinear_setup):
    """Tests assimilation using a more complex forward operator"""
    B, R, H = nonlinear_setup
    da = ThreeDVar(B, R, H)
    
    xb = np.array([1.0, 2.0])
    y = np.array([1.2, 4.5])
    
    xa, res = da.assimilate(xb, y)
    
    assert res.success is True

    # Resulting analysis should lie somewhere between the background and observation vectors
    assert np.logical_and(xa[0] > 1.0, xa[0] < 1.2) 
    assert np.logical_and(xa[1] > 2.0, xa[1] < 4.5)


def test_assimilate_linear_matrix_path():
    """Tests the special case where H is a matrix and solved analytically."""

    B = np.array([[2, 0.0, 0.0], [0.0, 2, 0.0], [0.0, 0.0, 2]])
    R = np.array([[1, 0.0], [0.0, 1]])
    H_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    da = ThreeDVar(B, R, H_matrix)
    
    xb = np.array([1.0, 1.0, 1.0])
    y = np.array([2.0, 2.0])
    
    xa, res = da.assimilate(xb, y)
    
    # In the linear matrix case, the returned 'res' should be None
    assert res is None
    
    # Manual math check:
    # d = y - Hxb = [1 1]
    # S = HBH^T + R = [[3 0] [0 3]]
    # K = B H^T S^-1 = [[2/3 0] [0 2/3] [0 0]]
    # xa = xb + Kd = [5/3 5/3]
    expected_xa = np.array([5/3, 5/3, 1])
    
    assert_allclose(xa, expected_xa, rtol=1e-6)


"""
End test_variational_da.py
"""
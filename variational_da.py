"""
Simple Variational Data Assimilation Schemes

Initial code draft from Google Gemini

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize


#---------------------------------------------------------------------------------------------------
# Main Program
#---------------------------------------------------------------------------------------------------

class ThreeDVar:
    """
    An Object-Oriented implementation of 3D Variational Data Assimilation.
    """

    def __init__(self, B, R, H):
        """
        Initializes the 3D-Var system and caches matrix inverses.

        It is assumed that the state vector is length M and the observation vector is length N

        Inputs
        ------
        B : np.array
            Background error covariances matrix (shape: M x M)
        R : np.array
            Observation error covariance matrix (shape: N X N)
        H : function or np.array
            Observation operator 
            If function, takes a vector of length M and returns a vector of length N
            If np.array, has shape N x M

        """
        self.H = H
        self.B = B
        self.R = R
        
        # Cache the inverses upon initialization to save compute time
        # during sequential assimilation cycles.
        self.B_inv = np.linalg.inv(B)
        self.R_inv = np.linalg.inv(R)


    def _cost_function(self, x, xb, y):
        """
        Internal method to calculate the cost function J(x)
        
        Inputs
        ------
        x : np.array
            State vector (shape: M)
        xb : np.array
            Background state vector (shape: M)
        y : np.array
            Observation vector (shape: N)
        
        Returns
        -------
        Scalar cost function (J)

        """
        dx = x - xb
        dy = y - self.H(x)
        
        J_b = 0.5 * dx.T @ self.B_inv @ dx
        J_o = 0.5 * dy.T @ self.R_inv @ dy
        
        return J_b + J_o


    def _assimilate_linear(self, xb, y):
        """
        Internal method that computes the exact analytic solution using the Kalman Gain.
        
        Can only use this method if H is a matrix!

        Inputs
        ------
        xb : np.array
            Background state vector (shape: M)
        y : np.array
            Observation vector (shape: N)
        
        Returns
        -------
        xa : np.array
            Analysis vector (shape: M)

        """

        H_mat = self.H  # Rename for clarity since it's a matrix
        
        # Innovation
        d = y - H_mat @ xb
        
        # Portion of Kalman gain that is inverted
        S = H_mat @ self.B @ H_mat.T + self.R
        
        # Kalman Gain: K = B * H^T * S^-1
        # Using np.linalg.solve(S, H_mat) is more numerically stable than inv(S)
        K = self.B @ H_mat.T @ np.linalg.inv(S)
        
        # Analysis state: xa = xb + K*d
        xa = xb + K @ d

        return xa


    def assimilate(self, xb, y):
        """
        Runs the assimilation step for a given background and observation.

        Inputs
        ------
        xb : np.array
            Background state vector (shape: M)
        y : np.array
            Observation vector (shape: N)
        
        Returns
        -------
        xa : np.array
            Analysis vector (shape: M)
        res : tuple or None
            If H is a function, output from scipy.optimize.minimize()
            If H is a np.array, None

        """
        
        # SPECIAL CASE: If H is a matrix (not a callable function), solve analytically
        if not callable(self.H):
            return self._assimilate_linear(xb, y), None

        res = minimize(
            fun=self._cost_function,
            x0=xb,
            args=(xb, y),
            jac='2-point',
            method='BFGS'
        )
        
        if not res.success:
            print(f"Warning: Optimization failed. Message: {res.message}")
            
        return res.x, res


"""
End variational_da.py
"""
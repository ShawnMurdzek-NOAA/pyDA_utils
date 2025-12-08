"""
A Simple EnKF Implementation in Python

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np


#---------------------------------------------------------------------------------------------------
# EnKF Python Class
#---------------------------------------------------------------------------------------------------

class enkf_1ob():
    """
    Class for a simple EnKF that assimilates a single observation

    Currently only contains the serial Ensemble Square-Root Filter (EnSRF)

    Parameters
    ----------
    x_b : 2D np.array
        Background forecast vector. Dimensions: (fcst vars, ens members)
    y_0 : float
        Observations
    Hx_b : 1D np.array
        Model in observation space. Dimensions: ens members
    ob_var : float
        Observation error variance
    localize : 1D np.array or None, optional
        Array used for observation-space localization (set to None to turn off). PbHT is multipled 
        by localize prior to computing the Kalman gain. Note that HPbHT is not localized. 
        Dimensions: fcst vars
    
    A Note About Localization
    -------------------------
    This class uses observation-space localization (e.g., Houtekamer and Mitchell 2001, MWR; 
    Hamill et al. 2001, MWR, eqn 12-13; Lei et al. 2018, JAMES, eqn 2). The `localize` parameter 
    only localizes PbHT and represents the correlations between the various model gridpoints and
    the observation (thus, `localize` depends on the distance between the observation being 
    assimilated and all model gridpoints). Unlike the references cited above, HPbHT is not 
    localized. The localization of HPbHT represents the correlations between various observations
    (and is, therefore, dependent on distances between observations). Because only a single 
    observation is being assimilated, the correlation is 1 (i.e., only a diagonal element of the
    localization of HPbHT is retained when assimilating a single ob, and the localization matrix
    has all ones along the diagonal).

    References
    ----------
    Theory: Houtekamer and Mitchell (2001, MWR), Whitaker and Hamill et al. (2002, MWR)
    Algorithm: Vetra-Carvalho et al. (2018, Tellus)

    """

    def __init__(self, x_b, y_0, Hx_b, ob_var, localize=None):

        self.x_b = x_b
        self.y_0 = y_0
        self.Hx_b = Hx_b
        self.ob_var = ob_var
        self.local = localize

        self.m, self.N = x_b.shape  # m = number of model variables, N = ensemble size


    def _compute_x_b_mean_dev(self):
        """
        Compute ensemble mean and member deviations
        """

        if not hasattr(self, 'x_b_bar'):
            self.x_b_bar = np.mean(self.x_b, axis=1)
            self.x_b_dev = self.x_b - self.x_b_bar[:, np.newaxis]         
    

    def _compute_Hx_mean_dev(self):
        """
        Compute the mean and deviations of Hx

        Following the algorithm in appendix B of Vetra-Carvalho et al. (2018, Tellus)...
        H(x_b_bar) = mean(H(x_b))
        H(x_b_dev) = H(x_b) - mean(H(x_b))
        """

        if not hasattr(self, 'Hx_b_bar'):
            self.Hx_b_bar = np.mean(self.Hx_b)
            self.Hx_b_dev = self.Hx_b - self.Hx_b_bar
    

    def _compute_Kalman_gain(self):
        """
        Compute the Kalman gain

        Also include calculations for:
            P_b H^T: Covariance of the estimate of the observation from the ensemble with the background
            H P_b H^T: Variance of the estimate of the observation from the ensemble

        P_b H^T: Houtekamer and Mitchell (2001) eqn (2)
        H P_b H^T: Houtekamer and Mitchell (2001) eqn (3)
        Kalman gain: Whitaker and Hamill (2002) eqn (2)
        """

        if not hasattr(self, 'K'):
            self._compute_x_b_mean_dev()
            self._compute_Hx_mean_dev()
 
            # Compute PbHT and apply localization
            self.PbHT = np.inner(self.x_b_dev, self.Hx_b_dev) / (self.N - 1)
            if self.local is not None:
                self.PbHT = self.PbHT * self.local

            # Compute HPbHT
            self.HPbHT = np.sum((self.Hx_b - self.Hx_b_bar)**2) / (self.N - 1)

            # Compute Kalman gain
            self.K = self.PbHT / (self.HPbHT + self.ob_var)
    

    def EnSRF(self):
        """
        Compute the analysis mean and deviations using the EnSRF

        EnSRF factor (alpha) comes from Whitaker and Hamill (2002) eqn (13)

        Whitaker and Hamill (2002)
        """

        if not hasattr(self, 'x_a'):
            self._compute_Kalman_gain()
            self.alpha = 1 / (1 + np.sqrt(self.ob_var / (self.HPbHT + self.ob_var)))
            self.x_a_bar = self.x_b_bar + (self.K * (self.y_0 - self.Hx_b_bar))
            self.x_a_dev = self.x_b_dev - self.alpha * np.outer(self.K, self.Hx_b_dev)
            self.x_a = self.x_a_dev + self.x_a_bar[:, np.newaxis]


"""
End enkf.py
"""

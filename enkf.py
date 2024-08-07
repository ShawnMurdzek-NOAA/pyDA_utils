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
        Array used for localization (set to None to turn off). PbHT is multipled by localize prior
        to computing the Kalman gain. Dimensions: fcst vars
    
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


    def compute_x_b_mean_dev(self):
        """
        Compute ensemble mean and member deviations
        """

        if not hasattr(self, 'x_b_bar'):
            self.x_b_bar = np.mean(self.x_b, axis=1)
            self.x_b_dev = self.x_b - self.x_b_bar[:, np.newaxis]         
    

    def compute_Hx_mean_dev(self):
        """
        Compute the mean and deviations of Hx

        Following the algorithm in appendix B of Vetra-Carvalho et al. (2018, Tellus)...
        H(x_b_bar) = mean(H(x_b))
        H(x_b_dev) = H(x_b) - mean(H(x_b))
        """

        if not hasattr(self, 'Hx_b_bar'):
            self.Hx_b_bar = np.mean(self.Hx_b)
            self.Hx_b_dev = self.Hx_b - self.Hx_b_bar
    

    def compute_PbHT(self):
        """
        Compute P_b H^T

        Houtekamer and Mitchell (2001) eqn (2)
        """

        if not hasattr(self, 'PbHT'):
            self.compute_x_b_mean_dev()
            self.compute_Hx_mean_dev()
            dum = np.zeros(self.m)
            for i in range(self.N):
                dum = dum + (self.x_b_dev[:, i] * (self.Hx_b[i] - self.Hx_b_bar))
            self.PbHT = dum / (self.N - 1)

            # Apply localization
            if self.local is not None:
                self.PbHT = self.PbHT * self.local
    

    def compute_HPbHT(self):
        """
        Compute H P_b H^T

        Houtekamer and Mitchell (2001) eqn (3)
        """
  
        if not hasattr(self, 'HPbHT'):
            self.compute_Hx_mean_dev()
            self.HPbHT = np.sum((self.Hx_b - self.Hx_b_bar)**2) / (self.N - 1)


    def compute_Kalman_gain(self):
        """
        Compute the Kalman gain

        Whitaker and Hamill (2002) eqn (2)
        """

        if not hasattr(self, 'K'):
            self.compute_PbHT()
            self.compute_HPbHT()
            self.K = self.PbHT / (self.HPbHT + self.ob_var)
    

    def compute_EnSRF_factor(self):
        """
        Compute the square root factor for the EnSRF

        Whitaker and Hamill (2002) eqn (13)
        """

        if not hasattr(self, 'alpha'):
            self.compute_HPbHT()
            self.alpha = 1 / (1 + np.sqrt(self.ob_var / (self.HPbHT + self.ob_var)))


    def EnSRF(self):
        """
        Compute the analysis mean and deviations using the EnSRF

        Whitaker and Hamill (2002)
        """

        if not hasattr(self, 'x_a'):
            self.compute_x_b_mean_dev()
            self.compute_Hx_mean_dev()
            self.compute_Kalman_gain()
            self.compute_EnSRF_factor()
            self.x_a_bar = self.x_b_bar + (self.K * (self.y_0 - self.Hx_b_bar))
            dum = np.zeros(self.x_b.shape)
            for i in range(self.N):
                dum[:, i] = self.x_b_dev[:, i] - (self.alpha * self.K * self.Hx_b_dev[i])
            self.x_a_dev = dum
            self.x_a = self.x_a_dev + self.x_a_bar[:, np.newaxis]


"""
End enkf.py
"""
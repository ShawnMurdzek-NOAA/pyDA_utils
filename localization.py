"""
Functions for DA Localization

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import geopy.distance as gd
from haversine import haversine_vector


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

def gaspari_cohn_5ord(dist, f):
    """
    Gaspari and Cohn (1999, QJRMS) eqn (4.10)

    Parameters
    ----------
    dist : np.array
        Distance between two items
    f : float
        Normalization factor (related to localization)
    
    Returns
    -------
    C : np.array
        Factor to reduce correlations by

    """

    # Do some initial check
    if type(dist) is not np.ndarray:
        raise ValueError("variable 'dist' must be an array")
    elif not np.all(dist >= 0):
        raise ValueError("variable 'dist' must be >= 0")
 
    # Initialize C as all zeros
    C = np.zeros(dist.shape)

    # First condition, dist <= f
    idx1 = (dist <= f)
    r1 = dist[idx1] / f
    C[idx1] = -0.25*(r1**5) + 0.5*(r1**4) + 0.625*(r1**3) - (5/3)*(r1**2) + 1

    # First condition, f < dist <= 2f
    idx2 = np.logical_and(dist > f, dist <= 2*f)
    r2 = dist[idx2] / f
    C[idx2] = (1/12)*(r2**5) - 0.5*(r2**4) + 0.625*(r2**3) + (5/3)*(r2**2) - 5*r2 + 4 - (2/3)/r2

    return C


class localization_fct():
    """
    Class that handles computing localization for EnKFs

    Parameters
    ----------
    fct : Python function
        Function to use for localization. Inputs: distance and localization distance
    
    """

    def __init__(self, fct):
        self.fct = fct


    def plot_localization_fct(self, local={1:{'ls':'-'}, 2:{'ls':'--'}, 3:{'ls':':'}}, 
                              ax=None, plot_kw={}, legend_kw={}):
        """
        Plot localization function for multiple localization values

        Parameters
        ----------
        local : dictionary, optional
            Key is the localization distance to plot, values are the plotting options
        ax : matplotlib.pyplot.axes, optional
            Axes to add plot to. Set to None to create a new axes
        plot_kw : dictionary, optional
            Additional keyword arguments passed to matplotlib.pyplot.plot()
        legend_kw : dictionary, optional
            Additional keyword arguments passed to matplotlib.pyplot.legend()
        
        Returns
        -------
        fig or ax : matplotlib.pyplot.figure or matplotlib.pyplot.axes
            Figure or axes containing the plot

        """
        
        # Create axes
        if ax is None:
            return_fig = True
            fig, ax = plt.subplots(nrows=1, ncols=1)
        
        # Make plot
        dist = np.linspace(0, max(local)*3, 200)
        for l in local.keys():
            ax.plot(dist, self.fct(dist, l), label=l, **local[l], **plot_kw)
        ax.legend(**legend_kw)
        ax.set_xlabel('distance')
        ax.set_ylabel('C')
    
        if return_fig:
            return fig
        else:
            return ax
    

    def compute_localization(self, model_coord, ob_coord, lv, lh):
        """
        Compute 1D localization array in horizontal and vertical directions

        Parameters
        ----------
        model_coord : np.array
            Model coordinates (height, lat, lon). Dimensions: (npts, 3)
        ob_coord : np.array
            Observation coordinates (height, lat, lon). Dimensions: (3)
        lv : float
            Vertical localization length
        lh : float
            Horizontal localization length (km)
        
        Returns
        -------
        C : np.array
            1D Localization array that can be multiplied by PbHT in an EnKF

        """

        Ch = self.compute_partial_localization(model_coord[:, 1:], ob_coord[1:], lh)
        Cv = self.compute_partial_localization(model_coord[:, 0], ob_coord[0], lv)

        return Ch*Cv


    def compute_partial_localization(self, model_coord, ob_coord, l):
        """
        Compute 1D localization array in either the horizontal or vertical directions

        Parameters
        ----------
        model_coord : np.array
            Model coordinates (lat, lon) or (height). Dimensions: (npts, 2) or (npts)
        ob_coord : np.array or float
            Observation coordinates (lat, lon) or (height). Dimensions: (2) or float
        l : float
            Localization length
        
        Returns
        -------
        C : np.array
            Localization array that can be multiplied by PbHT in an EnKF

        """

        # Assume coordinates are (lat, lon) if model_coord is 2D
        if len(model_coord.shape) == 1:
            dist = np.abs(model_coord - ob_coord)
        else:
            # Compute distances between (lat, lon) pts in km
            dist = self.compute_latlon_dist(model_coord, ob_coord)
        
        # Compute localization
        C = self.fct(dist, l)

        return C
    

    def compute_latlon_dist(self, coord_array, coord_pt, method='haversine'):
        """
        Compute the distance between an array of (lat, lon) points and a single (lat, lon) point

        Implementation uses the haversine function

        Parameters
        ----------
        coord_array : np.array
            Array of (lat, lon) coordinates. Dimensions: (npts, 2)
        coord_pt : float
            A single (lat, lon) point
        method : string, optional
            Method used to compute the distance. Options:
                'haversine' : Use the haversine function. Faster, but less accurate
                'geodesic' : Use geopy.distance. Slower, but more accurate
        
        Returns
        -------
        dist : np.array
            1D array of distances between coord_array and coord_pt (km)
        """
 
        if method == 'haversine':
            dist = np.squeeze(haversine_vector(coord_array, coord_pt, check=False, comb=True))
        elif method == 'geodesic':
            dist = np.zeros(coord_array.shape[0])
            for i in range(len(dist)):
                dist[i] = gd.distance(coord_array[i, :], coord_pt).km

        return dist


"""
End localization.py
"""

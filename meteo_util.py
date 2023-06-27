"""
General Meteorology Utilities

shawn.s.murdzek@noaa.gov
Date Created: 15 February 2023
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import metpy.calc as mc
from metpy.units import units


#---------------------------------------------------------------------------------------------------
# Define Functions
#---------------------------------------------------------------------------------------------------

def equil_vapor_prs(T):
    """
    Equilibrium vapor pressure with respect to liquid water

    Parameters
    ----------
    T : float
        Temperature (K)

    Returns
    -------
    es : float
        Equilibrium vapor pressure (Pa)

    Notes
    -----
    Equation is based on that from Bolton (1980) and reproduced in Markowski and Richardson (2010),
    eqn (2.16)

    """

    T = T - 273.15
    es = 611.2 * np.exp((17.67 * T) / (T + 243.5))

    return es


def equil_mix(T, p):
    """
    Equilibrium mixing ratio with respect to liquid water

    Parameters
    ----------
    T : float
        Temperature (K)
    p : float
        Pressure (Pa)

    Returns
    -------
    qvs : float
        Equilibrium mixing ratio (kg / kg)

    Notes
    -----
    Equation is based on Markowski and Richardson (2010), eqn (2.11)

    """

    epn = 0.622
    es = equil_vapor_prs(T)
    qvs = (epn * es) / (p - es)

    return qvs


def T_from_Tv(Tv, spfh, epsilon=0.622):
    """
    Computes temperature from the virtual temperature and specific humidity

    Parameters
    ----------
    Tv : float
        Virtual temperature (K)
    spfh : float
        Specific humidity (kg/kg)
    epsilon : float, optional
        Ratio of dry-air gas constant to water vapor gas constant

    Returns
    -------
    T : float
        Sensible temperature (K)

    """ 

    mix = mc.mixing_ratio_from_specific_humidity(spfh * units.kg / units.kg).magnitude
    T = Tv * epsilon * ((1. + mix) / (mix + epsilon))

    return T


"""
End meteo_util.py
"""

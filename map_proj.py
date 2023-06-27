"""
Functions for Map Projections

This code is based on the map_utils module in WRF, which can be found in the following WPS file:

WPS/geogrid/src/module_map_utils.F

Only the Lambert Conformal map projection (which is used for the HRRR) is currently supported.

Users may notice that (x, y) coordinates computed by the ll_to_xy_lc function below do not match
the grid indices in the UPP output files. These differences can exceed 1 km in some regions. The 
reason for this is NOT related to the Python code here. Instead, the problem is that the (lat, lon)
coordinates in UPP are slightly different than those found in the wrfout files. When these 
erroneous (lat, lon) coordinates are fed into ll_to_xy_lc, errors are introduced to the computed
(x, y) coordinates. There is a code snippet at the bottom of this file that computes (x, y) 
coordinates using the (lat, lon) coordinates from UPP and the (x, y) coordinates using the 
(lat, lon) coordinates from the wrfout files, and then compares these (x, y) coordinates to the
gridpoint indices, which is assumed to be the "truth". As shown in this code snippet, the (x, y)
coordinates computed using the wrfout (lat, lon) coordinates agree very well with the truth (MAE
of 0.0009 km), whereas the (x, y) coordinates computed using the UPP (lat, lon) coordinates do not
agree very well (MAE of 0.6448 km).

shawn.s.murdzek@noaa.gov
Date Created: 22 February 2023
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def ll_to_xy_lc(lat, lon, ref_lat=38.5, ref_lon=-97.5, truelat1=38.5, truelat2=38.5, 
                stand_lon=-97.5, dx=1., e_we=5400, e_sn=3180, knowni=2699, knownj=1589):
    """
    Convert (lat, lon) coordinates to (x, y) coordinates usig a Lambert Conformal map projection

    Parameters
    ----------
    lat : 1-D array
        Array of latitude coordinates in the range -90 to 90 deg N
    lon : 1-D array
        Array of longitude coordinates in the range -180 to 180 deg E
    ref_lat : float, optional
        Latitude coordinate of grid center (in range -90 to 90 deg N). Aka lat1.
    ref_lon : float, optional
        Longitude coordinate of grid center (in range -180 to 180 deg E). Aka lon1.
    truelat1 : float, optional
        First true latitude (in range -90 to 90 deg N)
    truelat2 : float, optional
        Second true latitude (in range -90 to 90 deg N). Can be the same as truelat1
    stand_lon : float, optional
        The longitude parallel to the y-axis (in range -180 to 180 deg E)
    dx : float, optional
        Grid spacing (km)
    e_we : integer, optional
        Number of points in x direction
    e_sn : integer, optional
        Number of points in y direction
    knowni : integer, optional
        Index of (ref_lon, ref_lat). If NaN, determine knowni using e_we
    knownj : integer, optional
        Index of (ref_lon, ref_lat). If NaN, determine knownj using e_sn

    Returns
    -------
    xi : array
        Array of x coordinates
    yi : array
        Array of y coordinates

    Notes
    -----
    The (x, y) "coordinates" are the decimal indices for the given (lat, lon) coordinates for the 
    specified map projection. To get coordinates in km, set dx = 1. The origin is assumed to be the
    SW corner.

    If knowni and knownj are specified, ref_lat and ref_lon do not necessarily need to be the center 
    of the grid as long as this coordinate aligns with knowni and knownj

    Default values for optional parameters come from the values used in the OSSE Nature Run (which
    is identical to the HRRR, except the gridspacing is reduced from 3 to 1 km) 

    Code is adapted from the llij_lc() subroutine in module_map_utils.F

    """

    # Set radius of the Earth (value comes from constants_module.F in WPS)
    re = 6370. 
    rebydx = re / dx

    # Compute deltalon
    deltalon = lon - stand_lon
    ind1 = np.where(deltalon > 180.)[0]
    ind2 = np.where(deltalon < -180.)[0]
    if len(ind1) > 0:
        deltalon[ind1] = deltalon[ind1] - 360.
    if len(ind2) > 0:
        deltalon[ind2] = deltalon[ind2] + 360.

    # Compute cosine of truelat1
    tl1r = np.deg2rad(truelat1)
    ctl1r = np.cos(tl1r)

    # Set hemisphere factor
    if truelat1 < 0:
        hemi = -1.
    else:
        hemi = 1.

    # Determine center gridpoint indices (from metgrid/src/process_domain_module.F)
    if np.isnan(knowni):
        knowni = 0.5 * e_we - 1
        knownj = 0.5 * e_sn - 1

    # Compute cone factor (round to 7 decimal places to match WRF)
    if np.isclose(truelat1, truelat2):
        cone = np.sin(np.abs(np.deg2rad(truelat1)))
    else:
        cone = np.log10(np.cos(np.deg2rad(truelat1))) - np.log10(np.cos(np.deg2rad(truelat2)))
        cone = cone / (np.log10(np.tan(np.deg2rad(45.-np.abs(truelat1)/2.))) -
                       np.log10(np.tan(np.deg2rad(45.-np.abs(truelat2)/2.))))

    # Find pole point
    deltalon1 = ref_lon - stand_lon
    if deltalon1 > 180:
        deltalon1 = deltalon1 - 360.
    elif deltalon1 < -180:
        deltalon1 = deltalon1 + 360.
    rsw = rebydx * (ctl1r/cone) * (np.tan(np.deg2rad(90.*hemi-ref_lat) / 2.) /
                                   np.tan(np.deg2rad(90.*hemi-truelat1) / 2.)) ** cone
    arg = cone * np.deg2rad(deltalon1)
    polei = hemi * knowni - (hemi * rsw * np.sin(arg))
    polej = hemi * knownj + (rsw * np.cos(arg))

    # Determine radii to each of the desired points
    rm = rebydx * (ctl1r/cone) * (np.tan(np.deg2rad(90.*hemi-lat) / 2.) /
                                  np.tan(np.deg2rad(90.*hemi-truelat1) / 2.)) ** cone 

    # Compute (xi, yi) values for each point (subtract 1 b/c Python uses 0 to represent the first 
    # index)
    arg = cone * np.deg2rad(deltalon)   
    xi = polei + (hemi * rm * np.sin(arg))
    yi = polej - (rm * np.cos(arg))

    # Flip xi, yi values in Southern Hemisphere
    xi = xi * hemi
    yi = yi * hemi

    return xi, yi


#---------------------------------------------------------------------------------------------------
# Plot Showing Differences Between WRF Gridpoint Locations and Calculated Gridpoint Locations
#---------------------------------------------------------------------------------------------------
'''
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fname = ['/mnt/lfs4/BMC/wrfruc/murdzek/nature_run_spring_v2/wrfnat_202204291200.grib2',
         '/mnt/lfs4/BMC/wrfruc/murdzek/wrf_grid.nc']
engine = ['pynio', None]
latname = ['gridlat_0', 'XLAT']
lonname = ['gridlon_0', 'XLONG']
grid_name = ['UPP', 'WRF']

# Open files
ds = []
for f, eng in zip(fname, engine):
    if engine == None:
        ds.append(xr.open_dataset(f))
    else:
        ds.append(xr.open_dataset(f, engine=eng))

# Plot differences
fig = plt.figure(figsize=(8, 10))
plt.subplots_adjust(left=0.01, bottom=0.02, right=0.92, top=0.95)
axes = []
xi2d, yi2d = [], []
for i, (d, latn, lonn, name) in enumerate(zip(ds, latname, lonname, grid_name)):  
    ax = fig.add_subplot(3, 1, i+1, projection=ccrs.PlateCarree())

    print('Plotting for %s' % name)

    # Compute gridpoint locations offline
    lat = np.squeeze(d[latn].values)
    lon = np.squeeze(d[lonn].values)
    shape = lat.shape
    xi, yi = ll_to_xy_lc(lat.ravel(), lon.ravel())
    xi2d.append(np.reshape(xi, shape))
    yi2d.append(np.reshape(yi, shape))

    # Compute "true" gridpoint locations
    xit, yit = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    diff = np.sqrt((xi2d[i] - xit)**2 + (yi2d[i] - yit)**2)

    cax = ax.pcolormesh(lon, lat, diff, vmin=0, vmax=1.25, transform=ccrs.PlateCarree())

    ax.set_title('%s $-$ Truth (MAE = %.4f km)' % (name, np.mean(diff)), size=20)
    ax.coastlines('50m')
    axes.append(ax)

# Plot difference between the two grids
print('Plotting difference between two grids')
ax = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
diff = np.sqrt((xi2d[0] - xi2d[1])**2 + (yi2d[0] - yi2d[1])**2)
cax = ax.pcolormesh(lon, lat, diff, vmin=0, vmax=1.25, transform=ccrs.PlateCarree())
ax.set_title('%s $-$ %s (MAE = %.5f km)' % (grid_name[0], grid_name[1], np.mean(diff)), size=20)
ax.coastlines('50m')
axes.append(ax)

cbar = plt.colorbar(cax, ax=axes, orientation='vertical', pad=0.02, aspect=30)
cbar.set_label('Cartesian coordinate differences (km)', size=16)

plt.savefig('gridpoint_diff.png')
plt.close()
'''

"""
End map_proj.py
"""

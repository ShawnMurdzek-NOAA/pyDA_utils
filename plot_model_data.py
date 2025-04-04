"""
Real-Data Model Plotting Class

CartoPy Note1: CartoPy downloads shapefiles to plot coastlines, state borders, etc. The package used
to do this does not work on Jet, so these files must be downloaded manually (the only way I've found
to do this is to download them onto my Mac and upload them to Jet via scp). These shapefiles can
be downloaded from the following website:

https://www.naturalearthdata.com/downloads/

and must be uploaded to the following directory, then unzipped::

~/.local/share/cartopy/shapefiles/natural_earth/<category>/

where <category> is specified in cfeature.NaturalEarthFeature (usually physical or cultural). 

CartoPy Note2: The projection used to create the axes (using `projection`) and the projection used
to plot the data (using `transform`). Do not need to be the same. `projection` is the projection 
used for plotting, whereas `transform` describes the projection of the underlying data. Because
(lat, lon) coordinates are always used, `transform` should always be PlateCarree(). More details
can be found here: https://stackoverflow.com/questions/42237802/plotting-projected-data-in-other-projectons-using-cartopy

Shawn Murdzek
shawn.s.murdzek@noaa.gov
Date Created: 4 October 2022
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
import xarray as xr
from metpy.plots import SkewT, Hodograph
import metpy.calc as mc
from metpy.units import units
import scipy.ndimage as sn

try:
    import wrf
except ImportError:
    print('cannot load WRF-python module')


#---------------------------------------------------------------------------------------------------
# Define Plotting Class
#---------------------------------------------------------------------------------------------------

class PlotOutput():
    """
    Class that handles plotting real-data model output from either WRF or FV3 as well as gridded
    observational data (e.g., Stage IV precip).

    Parameters
    ----------
    data : List of NetCDF4 filepointers (for WRF) or XArray DataSets (all other options)
        Actual data to plot
    dataset : string
        Dataset type ('wrf', 'fv3', 'upp', or 'stage4')
    fig : matplotlib.figure
        Figure that axes are added to
    nrows : integer
        Number of rows of subplots
    ncols : integer
        Number of columns of subplots
    axnum : integer
        Subplot number
    proj : CartoPy projection, optional
        Map projection used for plotting. Does not need to match the map projection of the data (the 
        map projection of the data is irrelevant because the coordinates used for plotting are 
        (lat, lon), which is not a projection).

    Notes
    -----

    'wrf' datasets can be opened using the following command with netCDF4:
    `nc.Dataset(filename)`

    'upp' and 'stage4' datasets can be opened using the following xarray command:
    `xr.open_dataset(filename, engine='pynio')`

    """

    def __init__(self, data, dataset, fig, nrows, ncols, axnum, proj=ccrs.LambertConformal()):

        self.outtype = dataset
        self.fig = fig
        self.nrows = nrows
        self.ncols = ncols
        self.n = axnum
        self.proj = proj

        # Dictionary used to hold metadata
        self.metadata = {}

        # Extract first dataset
        if self.outtype == 'wrf':
            self.fptr = data[0]
        elif self.outtype == 'fv3':
            raise ValueError('Raw FV3 output is not supported yet')
        elif (self.outtype == 'upp' or self.outtype == 'stage4'):
            self.ds = data[0]

        # Extract second dataset (if applicable)
        if len(data) > 1:
            if self.outtype == 'wrf':
                self.fptr2 = data[1]
            elif self.outtype == 'fv3':
                raise ValueError('Raw FV3 output is not supported yet')
            elif (self.outtype == 'upp' or self.outtype == 'stage4'):
                self.ds2 = data[1]


        # Extract time
        if (self.outtype == 'stage4' or self.outtype == 'upp'):
            sample = list(self.ds.keys())[0]
            itime = dt.datetime.strptime(self.ds[sample].attrs['initial_time'], '%m/%d/%Y (%H:%M)')
            # forecast time may or may not be in a list
            try:
                ftime = int(self.ds[sample].attrs['forecast_time'][0])
            except IndexError:
                ftime = int(self.ds[sample].attrs['forecast_time'])
            if self.ds[sample].attrs['forecast_time_units'] == 'hours':
                delta = dt.timedelta(hours=ftime)
            elif self.ds[sample].attrs['forecast_time_units'] == 'minutes':
                delta = dt.timedelta(minutes=ftime)
            elif self.ds[sample].attrs['forecast_time_units'] == 'days':
                delta = dt.timedelta(days=ftime)
            self.time = (itime + delta).strftime('%Y%m%d %H:%M:%S UTC')

            
    def _ingest_data(self, var, zind=[np.nan, np.nan], units=None, interp_field=None, interp_lvl=None, 
                     ptype='none0', diff=False, red_fct=None, smooth=False, gauss_sigma=5,
                     indices=[None, None]):
        """
        Extract a single variable to plot and interpolate if needed.

        Parameters
        ----------
        var : string
            Variable from model output file to plot
        zind : list of integers, optional
            Index in z-direction. Contains one value for each dataset 
        units : string, optional
            Units for var (WRF only)
        interp_field : string, optional
            Interpolate var to a surface with a constant value of interp_field (WRF only)
        interp_lvl : string, optional
            Value of constant-interp_field used during interpolaton (WRF only)
        ptype : string, optional
            Plot type. Used as the key to store metadata
        diff : boolean, optional
            Is this a difference plot?
        red_fct : function, optional
            Function for reduction in the z direction (e.g. np.amax, np.sum, etc.)
        smooth : boolean, optional
            Option to smooth output using a Gaussian filter
        gauss_sigma : float, optional
            standard deviation for Gaussian filter.
        indices : list, optional
            Indices to use for data when plotting. Has three levels. Outer level is the dataset
            (one element for single data, two element for difference plots). Intermediate level is
            the two horizontal dimensions. Inner level is the actual indices in [start, stop, step]
            format.

        Returns
        -------
        data : array
            An array containing var
        coords : list
            List of coordinates for data.
                For a horizontal cross section, this list is [lats, lons]
        ptype : string
            Plot type. Used as the key to store metadata

        """

        coords = []

        # Append an integer to ptype if already in use
        n = 1
        while ptype in self.metadata.keys():
            ptype = ptype[:-1] + str(n)
            n = n + 1
        self.metadata[ptype] = {}

        if self.outtype == 'wrf':

            # Extract raw variable from WRF output file
            if units != None:
                raw = wrf.getvar(self.fptr, var, units=units) 
            else: 
                raw = wrf.getvar(self.fptr, var)
            
            # Save metadata
            self.metadata[ptype]['var'] = var
            self.metadata[ptype]['name'] = raw.description
            self.metadata[ptype]['units'] = raw.units

            # Interpolate, if desired
            if interp_field != None:
                ifield = wrf.getvar(self.fptr, interp_field)
                data = wrf.interplevel(raw, ifield, interp_lvl)
                self.metadata[ptype]['interp'] = '%d-%s ' % (interp_lvl, ifield.units)
            else:
                data = raw
                self.metadata[ptype]['interp'] = ''
  
            # Get lat/lon coordinates
            lat, lon = wrf.latlon_coords(data)
            coords = [wrf.to_np(lat), wrf.to_np(lon)]

            # Extract time if not done so already
            # This should really be done in __init__, but it's a bit tricky finding the time in the 
            # NetCDF4 object
            if not hasattr(self, 'time'):
                self.time = np.datetime_as_string(data.Time.values)[:-10] + ' UTC'

            data = wrf.to_np(data)

        elif (self.outtype == 'upp' or self.outtype == 'stage4'):
            if not np.isnan(zind[0]):
                data = self.ds[var][zind[0], :, :]
            else:
                data = self.ds[var]

            # Save metadata
            self.metadata[ptype]['var'] = var
            self.metadata[ptype]['name'] = data.attrs['long_name']
            self.metadata[ptype]['units'] = data.attrs['units']
            self.metadata[ptype]['interp'] = ''

            # Get lat/lon coordinates
            coords = [self.ds['gridlat_0'].values, self.ds['gridlon_0'].values]
           
            # Perform reduction in vertical
            if red_fct != None:
                data = red_fct(data, axis=0)

            # Subset data using provided indices
            if indices[0] != None:
                data = data[indices[0][0][0]:indices[0][0][1]:indices[0][0][2],
                            indices[0][1][0]:indices[0][1][1]:indices[0][1][2]]
 
            # Create difference fields
            if diff:
                if not np.isnan(zind[1]):
                    data2 = self.ds2[var][zind[1], :, :]
                elif red_fct != None:
                    data2 = red_fct(self.ds2[var], axis=0)
                else:
                    data2 = self.ds2[var]
                if indices[1] != None:
                    data2 = data2[indices[1][0][0]:indices[1][0][1]:indices[1][0][2],
                                  indices[1][1][0]:indices[1][1][1]:indices[1][1][2]]
                if smooth:
                    data = sn.gaussian_filter(data, gauss_sigma)
                    data2 = sn.gaussian_filter(data2, gauss_sigma)
                data = data - data2

        if smooth and not diff:
            data = sn.gaussian_filter(data, gauss_sigma)
 
        return data, coords, ptype


    def _cfad_data(self, var, bins=[None], ptype='cfad0'):
        """
        Create bin counts for a Contoured Frequency by Altitude Diagram (CFAD)

        Parameters
        ----------
        var : string
            Field from file used to compute CFAD
        bins : Array or list of floats, optional
            List of left-most bin edges and the final right-most bin edges (so number of bins = 
            len(bins) - 1). If set to None, bins will span the minimum to maximum value of the data

        Returns
        -------
        cts : array of floats
            Array containing bin counts with dimensions (count, altitude)
        bin_edges : array of floats
            Bin edges for cfad

        """

        # Append an integer to ptype if already in use
        n = 1
        while ptype in self.metadata.keys():
            ptype = ptype[:-1] + str(n)
            n = n + 1
        self.metadata[ptype] = {}

        # Determine bin edges if bins == None
        if bins[0] == None:
            if self.outtype == 'upp':
                bins = np.linspace(self.ds[var].min(), self.ds[var].max(), 21)

        if self.outtype == 'upp':

            # Compute CFAD
            s = self.ds[var].shape
            cts = np.zeros([s[0], len(bins)-1])
            for i in range(s[0]):
                cts[i, :], bin_edges = np.histogram(self.ds[var][i, :, :], bins=bins)

            # Save metadata
            self.metadata[ptype]['var'] = var
            self.metadata[ptype]['name'] = self.ds[var].attrs['long_name']
            self.metadata[ptype]['units'] = self.ds[var].attrs['units']

        return cts, bin_edges, ptype


    def _closest_gpt(self, lon, lat):
        """
        Determine the indices for the model gridpoint closest to the given (lat, lon) coordinate

        Parameters
        ----------
        lon, lat : float
            (lat, lon) coordinate with units of (deg N, deg E)

        Returns
        -------
        i, j : integer
            Indices of the gridpoint closest to (lat, lon)

        """

        if self.outtype == 'wrf':
            lat2d = wrf.getvar(self.fptr, 'lat')
            lon2d = wrf.getvar(self.fptr, 'lon')
        elif self.outtype == 'upp':
            lat2d = self.ds['gridlat_0'].values
            lon2d = self.ds['gridlon_0'].values

        return np.unravel_index(np.argmin((lat2d - lat)**2 + (lon2d - lon)**2), lat2d.shape)


    def _create_hcrsxn_ax(self, data):
        """
        Add a matplotlib.axes instance to the figure for a horizontal cross sections

        Parameters
        ----------
        data : Array
            Output from _ingest_data() method

        """

        self.ax = self.fig.add_subplot(self.nrows, self.ncols, self.n, projection=self.proj) 


    def config_ax(self, coastlines=True, states=True, grid=True, scale='50m', line_kw={}):
        """
        Add cartopy features to plotting axes. Axes must be defined first.

        Parameters
        ----------
        coastlines : boolean, optional
            Option to add coastlines
        states : boolean, optional
            Option to add state borders
        grid : boolean, optional
            Option to add map grid
        scale : boolean, optional
            Scale of CartoPy features. Typical options are '10m', '50m', and '110m'
        line_kw : dictionary, optional
            Additional keyword arguments to pass to coastlines() and add_feature()

        """
        
        if len(line_kw) == 0:
            line_kw = {'linewidth':0.5, 'edgecolor':'k'}

        if coastlines:
            self.ax.coastlines(scale, **line_kw)

        if states:
            borders = cfeature.NaturalEarthFeature(category='cultural',
                                                   scale=scale,
                                                   facecolor='none',
                                                   name='admin_1_states_provinces')
            self.ax.add_feature(borders, **line_kw)

        if grid:
            self.ax.gridlines()


    def contourf(self, var, cbar=True, ingest_kw={}, cntf_kw={}, cbar_kw={}, label_kw={}):
        """
        Plot data using a filled contour plot

        Parameters
        ----------
        var : string
            Variable to plot
        cbar : boolean, optional
            Option to plot a colorbar
        ingest_kw : dict, optional
            Other keyword arguments passed to _ingest_data (key must be a string)
        cntf_kw : dict, optional
            Other keyword arguments passed to contourf (key must be a string)
        cbar_kw : dict, optional
            Other keyword arguments passed to colorbar (key must be a string)
        label_kw : dict, optional
            Other keyword arguments passed to colorbar.set_label (key must be a string)

        """

        data, coords, ptype = self._ingest_data(var, ptype='contourf0', **ingest_kw)

        if not hasattr(self, 'ax'):
            self._create_hcrsxn_ax(data)

        self.cax = self.ax.contourf(coords[1], coords[0], data, transform=ccrs.PlateCarree(), 
                                    **cntf_kw)

        if cbar:
            self.cbar = plt.colorbar(self.cax, ax=self.ax, **cbar_kw)
            self.cbar.set_label('%s%s (%s)' % (self.metadata[ptype]['interp'], 
                                               self.metadata[ptype]['name'], 
                                               self.metadata[ptype]['units']), **label_kw)


    def pcolormesh(self, var, cbar=True, ingest_kw={}, pcm_kw={}, cbar_kw={}, label_kw={}):
        """
        Plot data using pcolormesh

        Parameters
        ----------
        var : string
            Variable to plot
        cbar : boolean, optional
            Option to plot a colorbar
        ingest_kw : dict, optional
            Other keyword arguments passed to _ingest_data (key must be a string)
        cntf_kw : dict, optional
            Other keyword arguments passed to pcolormesh (key must be a string)
        cbar_kw : dict, optional
            Other keyword arguments passed to colorbar (key must be a string)
        label_kw : dict, optional
            Other keyword arguments passed to colorbar.set_label (key must be a string)

        """

        data, coords, ptype = self._ingest_data(var, ptype='pcolormesh0', **ingest_kw)

        if not hasattr(self, 'ax'):
            self._create_hcrsxn_ax(data)

        self.cax = self.ax.pcolormesh(coords[1], coords[0], data, transform=ccrs.PlateCarree(), 
                                      **pcm_kw)

        if cbar:
            self.cbar = plt.colorbar(self.cax, ax=self.ax, **cbar_kw)
            self.cbar.set_label('%s%s (%s)' % (self.metadata[ptype]['interp'], 
                                               self.metadata[ptype]['name'], 
                                               self.metadata[ptype]['units']), **label_kw)
    
    

    def plot_diff(self, var, ingest_kw={}, cntf_kw={}, cbar_kw={}, label_kw={}, auto=True):
        """
        Plot data using a filled contour plot

        Parameters
        ----------
        var : string
            Variable to plot
        ingest_kw : dict, optional
            Other keyword arguments passed to _ingest_data (key must be a string)
        cntf_kw : dict, optional
            Other keyword arguments passed to contourf (key must be a string)
        cbar_kw : dict, optional
            Other keyword arguments passed to colorbar (key must be a string)
        label_kw : dict, optional
            Other keyword arguments passed to colorbar.set_label (key must be a string)
        auto : boolean, optional
            Automatically use the 'bwr' colormap and scale the contour levels so they are centered
            on zero and include the max differences

        """

        data, coords, ptype = self._ingest_data(var, ptype='contourf0', diff=True, **ingest_kw)

        if not hasattr(self, 'ax'):
            self._create_hcrsxn_ax(data)

        if auto:
            mx = np.amax(np.abs(data))
            lvls = np.linspace(-mx, mx, 20) 
            self.cax = self.ax.contourf(coords[1], coords[0], data, lvls, 
                                        transform=ccrs.PlateCarree(), cmap='bwr', **cntf_kw)
        else:
            self.cax = self.ax.contourf(coords[1], coords[0], data, transform=ccrs.PlateCarree(), 
                                        **cntf_kw)

        # Compute RMSD
        rmsd = np.sqrt(np.mean(data*data))

        self.cbar = plt.colorbar(self.cax, ax=self.ax, **cbar_kw)
        self.cbar.set_label('diff %s%s (%s)\n[RMSD = %.2e]' % (self.metadata[ptype]['interp'], 
                                                               self.metadata[ptype]['name'], 
                                                               self.metadata[ptype]['units'],
                                                               rmsd), **label_kw)


    def contour(self, var, label=False, ingest_kw={}, cnt_kw={}, label_kw={}):
        """
        Plot data using contours

        Parameters
        ----------
        var : string
            Variable from model output file to plot
        label : boolean, optional
            Option to add contour labels
        ingest_kw : dict, optional
            Other keyword arguments passed to _ingest_data (key must be a string)
        cnt_kw : dict, optional
            Other keyword arguments passed to contour (key must be a string)
        cnt_kw : dict, optional
            Other keyword arguments passed to contour labels (key must be a string)

        """

        data, coords, ptype = self._ingest_data(var, ptype='contour0', **ingest_kw)

        if not hasattr(self, 'ax'):
            self._create_hcrsxn_ax(data)

        self.cax = self.ax.contour(coords[1], coords[0], data, transform=ccrs.PlateCarree(), 
                                   **cnt_kw)

        if label:
            all_txt = self.ax.clabel(self.cax, **label_kw)

            # Set pad to 0 so that inline_spacing can be used
            for txt in all_txt:
                txt.set_bbox(dict(boxstyle='square,pad=0', fc='none', ec='none'))

    
    def barbs(self, xvar, yvar, thin=1, ingest_kw={}, barb_kw={}):
        """
        Plot data using wind barbs

        Parameters
        ----------
        xvar, yvar : string
            Variables from model output file to plot
        thin : integer, optional
            Option to plot every nth barb
        ingest_kw : dict, optional
            Other keyword arguments passed to _ingest_data (key must be a string)
        barb_kw : dict, optional
            Other keyword arguments passed to barb (key must be a string)

        """

        xdata, coords, ptype = self._ingest_data(xvar, ptype='barb0', **ingest_kw)
        ydata, coords, ptype = self._ingest_data(yvar, ptype='barb0', **ingest_kw)

        self.metadata.pop('barb1')

        if not hasattr(self, 'ax'):
            self._create_hcrsxn_ax(xdata)

        self.cax = self.ax.barbs(coords[1][::thin, ::thin], coords[0][::thin, ::thin], 
                                 xdata[::thin, ::thin], ydata[::thin, ::thin], 
                                 transform=ccrs.PlateCarree(), **barb_kw)

    
    def quiver(self, xvar, yvar, thin=1, ingest_kw={}, qv_kw={}):
        """
        Plot data using vectors

        Parameters
        ----------
        xvar, yvar : string
            Variables from model output file to plot
        thin : integer, optional
            Option to plot every nth barb
        ingest_kw : dict, optional
            Other keyword arguments passed to _ingest_data (key must be a string)
        qv_kw : dict, optional
            Other keyword arguments passed to quiver (key must be a string)

        """

        xdata, coords, ptype = self._ingest_data(xvar, ptype='vector0', **ingest_kw)
        ydata, coords, ptype = self._ingest_data(yvar, ptype='vector0', **ingest_kw)

        self.metadata.pop('vector1')

        if not hasattr(self, 'ax'):
            self._create_hcrsxn_ax(xdata)

        self.cax = self.ax.quiver(coords[1][::thin, ::thin], coords[0][::thin, ::thin], 
                                  xdata[::thin, ::thin], ydata[::thin, ::thin], 
                                  transform=ccrs.PlateCarree(), **qv_kw)


    def plot(self, lon, lat, plt_kw={}):
        """
        Plot (lat, lon) coordinates

        Parameters
        ----------
        lon, lat : float
            Longitude and latitude coordinates to plot
        plt_kw : dict, optional
            Other keyword arguments passed to plot (key must be a string)

        """

        if not hasattr(self, 'ax'):
            self._create_hcrsxn_ax(None)

        self.cax = self.ax.plot(lon, lat, transform=ccrs.PlateCarree(), **plt_kw)


    def cfad(self, var, zvar, bins=None, prs=False, cntf_kw={}, cbar_kw={}, label_kw={}):
        """
        Plot data using a CFAD

        Parameters
        ----------
        var : string
            Variable to plot
        zvar : string
            Variable to use in the vertical. Must be 1D
        bins : None or list of floats, optional
            List of left-most bin edges and the final right-most bin edges (so number of bins = 
            len(bins) - 1). If set to None, bins will span the minimum to maximum value of the data
        prs : Boolean, optional
            Use pressure for vertical coordinate? If so, use a log scale 
        cntf_kw : dict, optional
            Other keyword arguments passed to contourf (key must be a string)
        cbar_kw : dict, optional
            Other keyword arguments passed to colorbar (key must be a string)
        label_kw : dict, optional
            Other keyword arguments passed to colorbar.set_label (key must be a string)

        """

        cts, bin_edges, ptype = self._cfad_data(var, bins=bins)
        bin_ctrs = bin_edges[:-1] + 0.5 * (bin_edges[1:] - bin_edges[:-1])

        if self.outtype == 'upp':
            x, y = np.meshgrid(bin_ctrs, self.ds[zvar])
            ylabel = '%s (%s)' % (self.ds[zvar].attrs['long_name'], self.ds[zvar].attrs['units'])

        self.ax = self.fig.add_subplot(self.nrows, self.ncols, self.n)

        if prs:
            self.cax = self.ax.contourf(x, np.log10(y), cts, **cntf_kw)
            self.ax.set_yticks(np.log10(y[::3, 0]))
            self.ax.set_yticklabels(['%.0f' % p for p in y[::3, 0]])
            self.ax.set_ylim([np.log10(y).max(), np.log10(y).min()])
        else:
            self.cax = self.ax.contourf(x, y, cts, **cntf_kw)

        self.cbar = plt.colorbar(self.cax, ax=self.ax, **cbar_kw)
        self.cbar.set_label('counts', **label_kw)
        self.ax.set_xlabel('%s (%s)' % (self.metadata[ptype]['name'], 
                                        self.metadata[ptype]['units']))
        self.ax.set_ylabel(ylabel)


    def skewt(self, lon, lat, hodo=True, barbs=True, thin=5, hodo_range=50., skew=None, 
              hodo_ax=None, bgd_lw=0.75, Tplot_kw={'linewidth':2.5, 'color':'r'}, 
              TDplot_kw={'linewidth':2.5, 'color':'b'}, Hplot_kw={'linewidth':2},
              fields={'PRES':'PRES_P0_L105_GLC0',
                      'TMP':'TMP_P0_L105_GLC0',
                      'SPFH':'SPFH_P0_L105_GLC0',
                      'UGRD':'UGRD_P0_L105_GLC0',
                      'VGRD':'VGRD_P0_L105_GLC0',
                      'HGT':'HGT_P0_L105_GLC0',
                      'SFC_HGT':'HGT_P0_L1_GLC0'}):
        """
        Plot a Skew-T, log-p diagram for the gridpoint closest to (lat, lon)

        Parameters
        ----------
        lon, lat : float
            Longitude and latitude coordinates for Skew-T (Skew-T is plotted for model gridpoint
            closest to this coordinate)
        hodo : boolean, optional
            Option to plot hodograph inset
        barbs : boolean, optional
            Option to plot wind barbs
        thin : integer, optional
            Plot every x wind barb, where x = thin
        hodo_range : float, optional
            Hodograph range (m/s)
        skew : metpy.plots.SkewT object, optional
            SkewT object to plot sounding on. Set to None to create a new SkewT object
        hodo_ax : metpy.plots.Hodograph object, optional
            Hodograph object to plot hodograph on. Set to None to create a new subset axes
        bgd_lw : float, optional
            Linewidth for background lines (dry adiabats, moist adiabats, mixing lines)
        Tplot_kw : dict, optional
            Other keyword arguments passed to pyplot.plot when plotting temperature (key must be a string)
        TDplot_kw : dict, optional
            Other keyword arguments passed to pyplot.plot when plotting dewpoint (key must be a string)
        Hplot_kw : dict, optional
            Other keyword arguments passed to pyplot.plot when plotting hodograph (key must be a string)
        fields : dictionary, optional
            Names of the fields used for plotting. Includes...
                PRES = Pressure field name (Pa)
                TMP = Temperature field name (K)
                SPFH = Specific humidity field name (unitless)
                UGRD = Zonal wind field name (m/s)
                VGRD = Meridional wind field name (m/s)
                HGT = Height field name (m MSL)
                SFC_HGT = Model surface height field name (m MSL)

        """

        # Determine indices closest to (lat, lon) coordinate
        i, j = self._closest_gpt(lon, lat)

        # Extract variables
        if self.outtype == 'wrf':
            p = wrf.getvar(self.fptr, 'p', units='mb')[:, i, j] 
            T = wrf.getvar(self.fptr, 'temp', units='degC')[:, i, j] 
            Td = wrf.getvar(self.fptr, 'td', units='degC')[:, i, j] 
            if (barbs or hodo):
                u = wrf.getvar(self.fptr, 'ua', units='m s-1')[:, i, j]
                v = wrf.getvar(self.fptr, 'va', units='m s-1')[:, i, j]
            if hodo:
                z = wrf.getvar(self.fptr, 'height_agl', units='m')[:, i, j]
            time = np.datetime_as_string(p.Time.values)[:-10] + ' UTC:\n'
        elif self.outtype == 'upp':
            p = self.ds[fields['PRES']][:, i, j] * 1e-2
            T = self.ds[fields['TMP']][:, i, j] - 273.15
            Td = mc.dewpoint_from_specific_humidity(p.values*units.hPa,
                                                    T.values*units.degC,
                                                    self.ds[fields['SPFH']][:, i, j].values).to('degC').magnitude
            if (barbs or hodo):
                u = self.ds[fields['UGRD']][:, i, j]
                v = self.ds[fields['VGRD']][:, i, j]
            if hodo:
                z = self.ds[fields['HGT']][:, i, j] - self.ds[fields['SFC_HGT']][i, j]
            time = self.time

        # Create figure
        if skew == None:
            self.skew = SkewT(self.fig, rotation=45)
            self.skew.ax.set_xlim(-40, 60)
            self.skew.ax.set_ylim(1000, 100)
        else:
            self.skew = skew

        self.skew.plot(p, T, **Tplot_kw)        
        self.skew.plot(p, Td, **TDplot_kw)        

        self.skew.plot_dry_adiabats(linewidth=bgd_lw)
        self.skew.plot_moist_adiabats(linewidth=bgd_lw)
        self.skew.plot_mixing_lines(linewidth=bgd_lw)

        if hodo:

            # Create hodograph axes
            if hodo_ax == None:
                hod = inset_axes(self.skew.ax, '35%', '35%', loc=1) 
                self.h = Hodograph(hod, component_range=hodo_range)
                if hodo_range <= 10:
                    increment = 2
                elif hodo_range <= 25:
                    increment = 5
                else:
                    increment = 10
                self.h.add_grid(increment=increment) 
            else:
                self.h = hodo_ax
 
            # Color-code hodograph based on height AGL
            zbds = [0, 1000, 3000, 6000, 9000]
            colors = ['k', 'r', 'b', 'g']
            for zi, zf, c in zip(zbds[:-1], zbds[1:], colors):
                ind = np.where(np.logical_and(z >= zi, z < zf))[0]
                ind = np.append(ind, ind[-1]+1)
                self.h.plot(u[ind], v[ind], c=c, **Hplot_kw)
            ind = np.where(z >= zbds[-1])[0]
            self.h.plot(u[ind], v[ind], c='goldenrod', linewidth=2) 

        if barbs:
            imax = np.where(p < 100)[0][0]
            self.skew.plot_barbs(p[:imax:thin], u[:imax:thin], v[:imax:thin])

        # Add title
        if skew == None:
            loc = '(%.3f $^{\circ}$N, %.3f $^{\circ}$E)' % (lat, lon)
            if (hodo or barbs):
                ttl = ('%s\n' % time) + r'$T$ ($^{\circ}$C), $T_{d}$ ($^{\circ}$C), wind (m s$^{-1}$) at %s' % loc
            else:
                ttl = ('%s\n' % time) + r'$T$ ($^{\circ}$C), $T_{d}$ ($^{\circ}$C) at %s' % loc
            self.skew.ax.set_title(ttl)
 

    def set_lim(self, minlat, maxlat, minlon, maxlon):
        """
        Set plotting limits

        Parameters
        ----------
        minlat, maxlat : float
            Latitude limits
        minlon, maxlon : float
            Longitude limits

        """

        self.ax.set_extent([minlon, maxlon, minlat, maxlat])


    def ax_title(self, txt='', **kwargs):
        """
        Create a title for the axes

        Parameters
        ----------
        txt : string, optional
            Text to add to the beginning of the title

        """

        s = '%s %s' % (txt, self.time)
        for k in self.metadata.keys():
            if (k[:-1] != 'contourf') and (k[:-1] != 'cfad') and (k[:-1] != 'pcolormesh'):
                s = s + '\n%s: %s%s (%s)' % (k, self.metadata[k]['interp'], 
                                             self.metadata[k]['name'], self.metadata[k]['units']) 

        self.ax.set_title(s, **kwargs)
        

"""
End plot_model_data.py
"""

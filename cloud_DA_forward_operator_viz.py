"""
Cloud Data Assimilation Forward Operator Visualization

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import sys
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#---------------------------------------------------------------------------------------------------
# Classes and Functions
#---------------------------------------------------------------------------------------------------

class sfc_cld_forward_operator_viz():
    """
    Manages a collection of methods for visualizing output from the ceilometer cloud forward operator

    Parameters
    ----------
    cfo_obj : cloud_DA_forward_operator.sfc_cld_forward_operator() object

    """

    def __init__(self, cfo_obj):
        self.cfo_obj = cfo_obj
    

    def _create_plot(self, ax, figsize=(8, 8), subplot_kw={}):
        """
        Generates a figure and axes if neither exist

        Parameters
        ----------
        ax : matplotlib.axes
            Axes object. Set to None to generate a new axes
        figsize : tuple, optional
            Figure size, by default (8, 8)
        subplot_kw : dict, optional
            Additional keywords passed to matplotlib.pyplot.add_subplot(), by default {}

        Returns
        -------
        fig : matplotlib.figure
            Matplotlib figure object
        ax : matplotlib.axes
            Matplotlib axes object
        return_fig : boolean
            True if a new figure was generated, otherwise False

        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, **subplot_kw)
            return_fig = True
        else:
            fig = None
            return_fig = False
        return fig, ax, return_fig


    def scatterplot(self, x='ob_cld_amt', y='hofx', ax=None, scatter_kwargs={}, one_to_one=True):
        """
        Generate a scatterplot. Default is to plot observed cloud amount vs H(x)

        Parameters
        ----------
        x : str, optional
            Variable to plot on x-axis (must be in cfo_obj.data), by default 'ob_cld_amt'
        y : str, optional
            Variable to plot on x-axis (must be in cfo_obj.data), by default 'hofx'
        ax : matplotlib.axes, optional
            Axes to add plot to, by default None (which generates a new axes)
        scatter_kwargs : dict, optional
            Keyword arguments passed to matplotlib.pyplot.scatter, by default {}
        one_to_one : bool, optional
            Option to plot 1-to-1 line, by default True

        Returns
        -------
        ax (matplotlib.axes) or fig (matplotlib.figure)
            Axes or figure object containing the plot

        """

        fig, ax, return_fig = self._create_plot(ax)
        
        # Create plot
        xval = self.cfo_obj.flatten1d(x)
        yval = self.cfo_obj.flatten1d(y)
        ax.scatter(xval, yval, **scatter_kwargs)
        ax.set_xlabel(x, size=14)
        ax.set_ylabel(y, size=14)
        ax.grid()
        if one_to_one:
            minval = max([np.amin(xval), np.amin(yval)])
            maxval = max([np.amax(xval), np.amax(yval)])
            ax.plot([minval, maxval], [minval, maxval], 'k-')

        if return_fig:
            return fig
        else:
            return ax


    def hist(self, ax=None, plot_param={'field':'OmB', 'xlabel':'O $-$ B (cloud fraction, %)'},
             hist_kwargs={'bins':17, 'edgecolor':'k', 'linewidth':1}):
        """
        Generate a 1-D histogram

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axes to add plot to, by default None (which generates a new axes)
        plot_param : dict, optional
            Plotting parameters, by default {'field':'OmB', 'xlabel':'O $-$ B (cloud fraction, %)'}
        hist_kwargs : dict, optional
            Keyword arguments passed to matplotlib.pyplot.hist, by default {'bins':17, 'edgecolor':'k', 'linewidth':1}

        Returns
        -------
        ax (matplotlib.axes) or fig (matplotlib.figure)
            Axes or figure object containing the plot

        """

        fig, ax, return_fig = self._create_plot(ax)

        # Check to ensure that plot_param['field'] exists
        if plot_param['field'] not in self.cfo_obj.data:
            print(f"sfc_cld_forward_operator_viz.hist: Field does not exist: {plot_param['field']}")

        # Create plot
        ax.hist(self.cfo_obj.flatten1d(plot_param['field']), **hist_kwargs)
        ax.set_xlabel(plot_param['xlabel'], size=14)
        ax.set_ylabel('count', size=14)
        ax.grid(axis='y')

        if return_fig:
            return fig
        else:
            return ax
    

    def _reformat_arrays(self, field, idx, zfield='model_col_height_agl'):
        """
        Reformat model data from cfo_obj.data into 2D arrays with dimensions (height, ob index)

        Parameters
        ----------
        field : string
            Model field from cfo_obj.data to reformat
        idx : tuple-like
            Obs indices to include in the reformatted data
        zfield : str, optional
            Model height field from cfo_obj.data, by default 'model_col_height_agl'

        Returns
        -------
        idx_2d : np.array
            Obs indices from cfo_obj.data['idx']. Dimensions (height, ob index)
        hgt_2d : np.array
            Height array. Dimensions (height, ob index)
        field_2d : np.array
            Data from cfo_obj.data[field]. Dimensions (height, ob index)

        """

        zdim = max([len(self.cfo_obj.data[zfield][i]) for i in idx])
        idx_2d = np.zeros([zdim, len(idx)])
        hgt_2d = np.zeros([zdim, len(idx)])
        field_2d = np.ones([zdim, len(idx)]) * np.nan
        for i, ix in enumerate(idx):
            idx_2d[:, i] = ix
            zlen = len(self.cfo_obj.data[zfield][ix])
            hgt_2d[:zlen, i] = self.cfo_obj.data[zfield][ix]
            hgt_2d[zlen:, i] = hgt_2d[zlen-1, i] + (hgt_2d[zlen-1, i] - hgt_2d[zlen-2, i])
            field_2d[:zlen, i] = self.cfo_obj.data[field][ix]
        return idx_2d, hgt_2d, field_2d
    

    def _compute_cell_edges(self, hgt_2d):
        """
        Compute 2D arrays of cell edges from 2D arrays of cell centers.
        Note that the cell edge arrays increase the lengths of each of the two original dimensions by 1

        Parameters
        ----------
        hgt_2d : np.array
            Model height cel centers as a 2D array

        Returns
        -------
        idx_edges : np.array
            2D array of ob index edges (e.g., -0.5, 0.5, 1.5, etc.)
        hgt_edges : np.array
            2D array of height edges. Linear interpolation is used between height cell centers from 
            adjacent ob indices

        """
        ni, nj = np.shape(hgt_2d)
        idx_edges = np.zeros([ni+1, nj+1])
        hgt_edges = np.zeros([ni+1, nj+1])

        for i in range(ni+1):
            idx_edges[i, :] = np.arange(-0.5, nj)

        hgt_edges[1:-1, 0] = 0.5*(hgt_2d[1:, 0] + hgt_2d[:-1, 0])
        hgt_edges[1:-1, -1] = 0.5*(hgt_2d[1:, -1] + hgt_2d[:-1, -1])
        hgt_edges[1:-1, 1:-1] = 0.25*(hgt_2d[1:, 1:] + hgt_2d[1:, :-1] + hgt_2d[:-1, 1:] + hgt_2d[:-1, :-1])
        hgt_edges[0, :] = hgt_edges[1, :] - (hgt_edges[2, :] - hgt_edges[1, :])
        hgt_edges[-1, :] = hgt_edges[-2, :] + (hgt_edges[-2, :] - hgt_edges[-3, :])
        hgt_edges[hgt_edges < 0] = 0
        
        return idx_edges, hgt_edges


    def _reformat_pts(self, field, idx, zfield='HOCB'):
        """
        Create 1D arrays of ob index, height, and another specified field

        Parameters
        ----------
        field : string
            Observation field from cfo_obj.data to create a 1D array for
        idx : tuple-like
            Observation indices to include
        zfield : str, optional
            Field from cfo_obj.data corresponding to ob height, by default 'HOCB'

        Returns
        -------
        1D arrays for ob index, ob height, and ob field

        """

        idx_list = []
        field_list = []
        hgt_list = []
        for i in idx:
            for hgt, val in zip(self.cfo_obj.data[zfield][i], 
                                self.cfo_obj.data[field][i]):
                idx_list.append(i)
                hgt_list.append(hgt)
                field_list.append(val)
        return np.array(idx_list), np.array(hgt_list), np.array(field_list)


    def vert_columns(self, ax=None, idx=list(range(10)), zlim=[0, 3500],
                     pcolor_param={'field':'model_col_TCDC_P0_L105_GLC0', 
                                   'label':'Model Cloud Amount (%)',
                                   'kwargs':{'vmin':0, 'vmax':100, 'cmap':'plasma_r'}},
                     pt_param={'field':'ob_cld_amt', 
                               'label':'Ob Cloud Amount (%)', 
                               'kwargs':{'vmin':0, 'vmax':100, 's':75, 'edgecolors':'k', 'cmap':'plasma_r'}}):
        """
        Generate a 2D plot showing vertical columns above each ceilometer, with one field plotted 
        using pcolor and the other using plot

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axes object to add plot to, by default None (which generates a new axes)
        idx : tuple-like, optional
            Ceilometer indices from cfo_obj.data['idx'] to plot (note this differs from SID), 
            by default list(range(10))
        zlim : list, optional
            Vertical range (m), by default [0, 3500]
        pcolor_param : dict, optional
            Plotting parameters for matplotlib.pyplot.pcolor, by default 
            {'field':'model_col_TCDC_P0_L105_GLC0', 'label':'Model Cloud Amount (%)', 'kwargs':{'vmin':0, 'vmax':100, 'cmap':'plasma_r'}}
        pt_param : dict, optional
            Plotting parameters for matplotlib.pyplot.plot, by default 
            {'field':'ob_cld_amt', 'label':'Ob Cloud Amount (%)', 'kwargs':{'vmin':0, 'vmax':100, 's':75, 'edgecolors':'k', 'cmap':'plasma_r'}}

        Returns
        -------
        ax (matplotlib.axes) or fig (matplotlib.figure)
            Axes or figure object containing the plot

        """

        fig, ax, return_fig = self._create_plot(ax, figsize=(12, 8))

        # Reformat points for plotting
        # Use cell edges, otherwise pcolor throws a warning. Note that (idx_2d, hgt_2d) are cell centers
        idx_2d, hgt_2d, field_2d = self._reformat_arrays(pcolor_param['field'], idx)
        idx_edges, hgt_edges = self._compute_cell_edges(hgt_2d)
        idx_pt, hgt_pt, field_pt = self._reformat_pts(pt_param['field'], idx)
        xpt = np.zeros(len(idx_pt))
        tick_loc = []
        tick_label = []
        for n, i in enumerate(idx_2d[0, :]):
            xpt[idx_pt == i] = n
            tick_loc.append(n)
            tick_label.append(i)

        # Plot filled colors. 
        cax_2d = ax.pcolor(idx_edges, hgt_edges, field_2d, **pcolor_param['kwargs'])
        cbar_2d = plt.colorbar(cax_2d, ax=ax)
        cbar_2d.set_label(pcolor_param['label'], size=14)

        # Plot points
        cax_pt = ax.scatter(xpt, hgt_pt, c=field_pt, **pt_param['kwargs'])
        cbar_pt = plt.colorbar(cax_pt, ax=ax)
        cbar_pt.set_label(pt_param['label'], size=14)

        # Add axes labels and plot limits
        ax.set_xlabel('Index', size=14)
        ax.set_ylabel('Height (m AGL)', size=14)
        ax.set_ylim(zlim)
        try:
            ax.set_xticks(tick_loc, labels=np.array(tick_label, dtype=int))
        except TypeError:
            None

        if return_fig:
            return fig
        else:
            return ax
    

    def composite_cld_cover(self, ax=None, proj=ccrs.LambertConformal(), map_scale='50m', 
                            lon_lim=[-124, -70], lat_lim=[21, 49],
                            pcolor_param={'field':'TCDC_P0_L105_GLC0', 
                                          'label':'Model Cloud Amount (%)',
                                          'kwargs':{'vmin':0, 'vmax':100, 'cmap':'plasma_r'}},
                            pt_param={'field':'ob_cld_amt', 
                                      'label':'Ob Cloud Amount (%)', 
                                      'kwargs':{'vmin':0, 'vmax':100, 's':40, 'edgecolors':'k', 'cmap':'plasma_r'}}):
        """
        Generate a plot of composite cloud cover (i.e., max cloud fraction in a vertical column) 
        with point observations overlaid

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axes object to add plot to, by default None (which generates a new axes)
        proj : cartopy.crs, optional
            Map projection, by default ccrs.LambertConformal()
        map_scale : str, optional
            Map scale, by default '50m'
        lon_lim : tuple-like, optional
            Longitude plotting limits (deg E), by default [-124, -70]
        lat_lim : tuple-like, optional
            Latitude plotting limits (deg N), by default [21, 49]
        pcolor_param : dict, optional
            Plotting parameters for matplotlib.pyplot.pcolor, by default 
            {'field':'model_col_TCDC_P0_L105_GLC0', 'label':'Model Cloud Amount (%)', 'kwargs':{'vmin':0, 'vmax':100, 'cmap':'plasma_r'}}
        pt_param : dict, optional
            Plotting parameters for matplotlib.pyplot.plot, by default 
            {'field':'ob_cld_amt', 'label':'Ob Cloud Amount (%)', 'kwargs':{'vmin':0, 'vmax':100, 's':40, 'edgecolors':'k', 'cmap':'plasma_r'}}

        Returns
        -------
        ax (matplotlib.axes) or fig (matplotlib.figure)
            Axes or figure object containing the plot

        """
        
        fig, ax, return_fig = self._create_plot(ax, figsize=(10, 8), subplot_kw={'projection':proj})

        # Plot model field
        cax_model = ax.pcolormesh(self.cfo_obj.model_ds['gridlon_0'], self.cfo_obj.model_ds['gridlat_0'],
                                  np.amax(self.cfo_obj.model_ds[pcolor_param['field']], axis=0),
                                  transform=ccrs.PlateCarree(),
                                  **pcolor_param['kwargs'])
        cbar_2d = plt.colorbar(cax_model, ax=ax, orientation='horizontal')
        cbar_2d.set_label(pcolor_param['label'], size=14)

        # Plot obs
        xpt = np.zeros(len(self.cfo_obj.data['idx']))
        ypt = np.zeros(len(self.cfo_obj.data['idx']))
        ob_cld_frac = np.zeros(len(self.cfo_obj.data['idx']))
        for n, i in enumerate(self.cfo_obj.data['idx']):
            xpt[n] = self.cfo_obj.data['lon'][i]
            ypt[n] = self.cfo_obj.data['lat'][i]
            ob_cld_frac[n] = np.amax(self.cfo_obj.data['ob_cld_amt'][i])
        cax_pt = ax.scatter(xpt, ypt, c=ob_cld_frac, transform=ccrs.PlateCarree(), **pt_param['kwargs'])
        cbar_pt = plt.colorbar(cax_pt, ax=ax, orientation='horizontal')
        cbar_pt.set_label(pt_param['label'], size=14)

        # Add country and state borders
        ax.coastlines(map_scale, linewidth=0.5, edgecolor='k')
        borders = cfeature.NaturalEarthFeature(category='cultural',
                                               scale=map_scale,
                                               facecolor='none',
                                               name='admin_1_states_provinces')
        ax.add_feature(borders, linewidth=0.5, edgecolor='gray')
        ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])

        if return_fig:
            return fig
        else:
            return ax


"""
End cloud_DA_forward_operator_viz.py
"""
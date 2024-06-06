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

    def __init__(self, cfo_obj):
        self.cfo_obj = cfo_obj
    

    def _create_plot(self, ax, figsize=(8, 8), subplot_kw={}):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, **subplot_kw)
            return_fig = True
        else:
            fig = None
            return_fig = False
        return fig, ax, return_fig


    def scatterplot(self, x='ob_cld_amt', y='hofx', ax=None, scatter_kwargs={}, one_to_one=True):

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
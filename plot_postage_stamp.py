"""
Functions to Create Postage Stamp Plots Using plot_model_data.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pyDA_utils.plot_model_data as pmd

try:
    import netCDF4 as nc
except ImportError:
    None
try:
    import xarray as xr
except ImportError:
    None

#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

class PlotPostageStamp():
    """
    Class that handles creating postage stamp plots for ensemble output

    Parameters
    ----------
    fnames : List of lists, with each sublist containing 1-2 strings
        Filenames to plot
        Each sublist represents a different ensemble member. Each sublist contains either a file
        from a single ensemble, or two files from two different ensembles. In the latter case, 
        difference plots are created.
    names : List of strings
        Ensemble member names
    dataset : string
        Dataset type ('wrf', ''upp', or 'stage4')
    nrows : integer
        Number of rows of subplots
    ncols : integer
        Number of columns of subplots
    fig_kw : dictionary, optional
        Keyword arguments to pass to plt.figure() function
    io_kw : dictionary, optional
        Keyword arguments to pass to either nc.Dataset or xr.open_dataset
    proj : CartoPy projection, optional
        Map projection used for plotting. Does not need to match the map projection of the data (the
        map projection of the data is irrelevant because the coordinates used for plotting are
        (lat, lon), which is not a projection). Set to None for idealized model output that does not
        have meaningful (lat, lon) coordinates.

    """

    def __init__(self, fnames, names, dataset, nrows, ncols, fig_kw={'figsize':(12, 12)}, io_kw={}, 
                 proj=ccrs.LambertConformal()):

        self.fnames = fnames
        self.names = names
        self.outtype = dataset
        self.nrows = nrows
        self.ncols = ncols
        self.proj = proj
        self.n = len(fnames)

        # Check that required modules are loaded
        if dataset in ['wrf']:
            try:
                nc.__version__
            except NameError:
                raise ImportError(f"Unable to load netCDF4, which is required for {dataset}")
        elif dataset in ['upp', 'stage4']:
            try:
                xr.__version__
            except NameError:
                raise ImportError(f"Unable to load xarray, which is required for {dataset}")
        else:
            raise ValueError(f"{dataset} is not a valid dataset type")

        # Create a figure for postage stamp plot
        self.fig = plt.figure(**fig_kw)

        # Create pmd.PlotOutput() objects for each file
        pmd_objs = []
        for i, fname_list in enumerate(fnames):
            tmp = []
            for f in fname_list:
                if dataset in 'wrf':
                    tmp.append(nc.Dataset(f, **io_kw))
                elif dataset in ['upp', 'stage4']:
                    tmp.append(xr.open_dataset(f, **io_kw))
            pmd_objs.append(pmd.PlotOutput(tmp, 
                                           dataset, 
                                           self.fig, 
                                           nrows, 
                                           ncols, 
                                           i+1,
                                           proj=proj))
        self.pmd_objs = pmd_objs


    def _clean_axes(self):
        """
        Clean up axes 

        """

        # Turn off tick labels for some subplots
        for i in range(self.n):
            if i%self.ncols != 0:
                self.pmd_objs[i].ax.yaxis.set_ticklabels([])
            if (self.n - i) > self.ncols:
                self.pmd_objs[i].ax.xaxis.set_ticklabels([])

        # Set aspect ratio to 1 for idealized simulations
        if self.proj is None:
            for i in range(self.n):
                self.pmd_objs[i].ax.set_aspect('equal')


    def contourf(self, var, cbar_all=True, kw={}, cbar_kw={}, label_kw={},
                 cbar_ax_lim=[0.91, 0.05, 0.03, 0.8]):
        """
        Create contourf plots

        Parameters
        ----------
        var : string
            Variable to plot
        cbar_all : boolean, optional
            Option to plot a single colorbar rather than 1 colorbar for each subplot
        kw : dict, optional
            Other keyword arguments passed to pmd.PlotOutput.contourf (key must be a string)
        cbar_kw : dict, optional
            Other keyword arguments passed to colorbar (key must be a string)
        label_kw : dict, optional
            Other keyword arguments passed to colorbar.set_label (key must be a string)
        cbar_ax_lim : list of floats
            Location of colorbar axes. Format: [left, bottom, width, height]

        """

        # Turn off colorbar for each individual subplot
        if cbar_all:
            kw['cbar'] = False

        # Check to see if axes have been created yet
        first_plot = False
        if not hasattr(self.pmd_objs[0], 'ax'):
            first_plot = True

        # Create plots
        for i in range(self.n):
            self.pmd_objs[i].contourf(var, **kw)

        # Add colorbar
        if cbar_all:
            smpl_obj = self.pmd_objs[0]
            cbar_ax = self.fig.add_axes(cbar_ax_lim)
            self.cbar = plt.colorbar(smpl_obj.cax, cax=cbar_ax, **cbar_kw)
            self.cbar.set_label('%s%s (%s)' % (smpl_obj.metadata['contourf0']['interp'],
                                               smpl_obj.metadata['contourf0']['name'],
                                               smpl_obj.metadata['contourf0']['units']), **label_kw)

        # Clean up axes
        if first_plot:
            self._clean_axes()


    def ax_titles(self, **kwargs): 
        """
        Add titles to each axes

        """

        for i, txt in enumerate(self.names):
            self.pmd_objs[i].ax.set_title(txt, **kwargs)


    def sup_title(self, **kwargs): 
        """
        Add an overall title to the figure

        """

        smpl_obj = self.pmd_objs[0]
        s = smpl_obj.time
        for k in smpl_obj.metadata.keys():
            if (k[:-1] != 'contourf') and (k[:-1] != 'cfad') and (k[:-1] != 'pcolormesh'):
                s = s + '\n%s: %s%s (%s)' % (k, smpl_obj.metadata[k]['interp'],
                                             smpl_obj.metadata[k]['name'], smpl_obj.metadata[k]['units'])
        plt.suptitle(s, **kwargs)


"""
End plot_postage_stamp.py
"""

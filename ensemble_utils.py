"""
Utilities Related to Examining Ensemble Output

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.interpolate as si
import pandas as pd
import copy

import pyDA_utils.plot_model_data as pmd
import pyDA_utils.bufr as bufr
import pyDA_utils.upp_postprocess as uppp


#---------------------------------------------------------------------------------------------------
# Define Ensemble Class
#---------------------------------------------------------------------------------------------------

class ensemble():
    """
    Class that handles the processing and plotting of ensemble output

    Parameters
    ----------
    fnames : dictionary of strings
        Ensemble UPP output files. Key is the member name.
    verbose : boolean, optional
        Option for verbose output
    extra_fnames : dictionary of strings, optional
        Extra ensemble UPP output files. Key is the member name
    extra_fields : list of strings, optional
        List of extra fields to add to the ensemble DataSets from the extra_fnames files. Set to an
        empty list to not use this feature.
    bufr_csv_fname : string, optional
        BUFR CSV file containing verifying obs. Set to None to not use
    lat_limits : list of floats, optional
        Latitude limits for subset (deg N)
    lon_limits : list of floats, optional
        Longitude limits for subset (deg E, use negative values for W. hemisphere)
    zind : list of integers, optional
        Vertical indices for subset
    zfield : string, optional
        Field to apply zind to. Set to None to not subset domain in vertical
    state_fields : list of strings, optional
        Fields to include in state matrix
    bec : boolean, optional
        Option to compute background error covariance (BEC) matrix

    """

    def __init__(self, fnames, verbose=True, extra_fnames={}, extra_fields=[], bufr_csv_fname=None,
                 lat_limits=[10, 53], lon_limits=[-134.1, -60.9], zind=list(range(1, 66)), 
                 zfield='lv_HYBL2', state_fields=[], bec=False):

        self.fnames = fnames
        self.mem_names = list(fnames.keys())        
        self.lat_limits = lat_limits
        self.lon_limits = lon_limits

        # Read in ensemble UPP output
        datasets = {}
        for key in self.mem_names:
            if verbose:
                print('reading member {name}'.format(name=key))
            datasets[key] = xr.open_dataset(fnames[key], engine='pynio')
            if len(extra_fields) > 0:
                extra_ds = xr.open_dataset(extra_fnames[key], engine='pynio')
                for f in extra_fields:
                    datasets[key][f] = extra_ds[f].copy()
        self.ds = datasets

        # Read in verifying obs
        if bufr_csv_fname != None:
            self.verif_obs = bufr.bufrCSV(bufr_csv_fname)

        # Subset ensemble output
        if len(self.ds) > 0:
            self.subset_ds = self._subset_ens_domain(lon_limits[0], lon_limits[1], lat_limits[0], 
                                                     lat_limits[1], zind, zfield=zfield)

        # Compute state matrix, ensemble statistics, and BEC matrix
        if len(state_fields) > 0:
            self.state_matrix = self._create_state_matrix(state_fields)
            self.state_matrix.update(self._compute_ens_stats())
            self.state_matrix['ens_dev'] = self._compute_ens_deviations()
            if bec:
               if verbose:
                   print('computing BEC...')
               self.state_matrix['be_cov'], self.state_matrix['be_corr'] = self._compute_bec()


    def _get_corner_ind(self, min_lon, max_lon, min_lat, max_lat, debug=False):
        """
        Get the corner indices for the subset domain

        Parameters
        ----------
        min_lon : float
            Minimum longitude (deg E, use negatives for W. hemisphere)     
        max_lon : float
            Maximum longitude (deg E, use negatives for W. hemisphere)     
        min_lat : float
            Minimum latitude (deg N)     
        max_lat : float
            Maximum latitude (deg N)     
        debug : boolean, optional
            Option to print extra output for debugging

        Returns
        -------
        i_llcrnr : integer
            i index for lower-left corner
        i_urcrnr : integer
            i index for upper-right corner
        j_llcrnr : integer
            j index for lower-left corner
        j_urcrnr : integer
            j index for upper-right corner

        """

        # Determine indices for the x and y dimensions by finding the indices of gridpoints closest to the corners
        iind = np.zeros(4, dtype=int)
        jind = np.zeros(4, dtype=int)
        n = 0
        tmp_ds = self.ds[self.mem_names[0]]
        for lon in [min_lon, max_lon]:
            for lat in [min_lat, max_lat]:
                iind[n], jind[n] = np.unravel_index(np.argmin((tmp_ds['gridlon_0'].values - lon)**2 + 
                                                              (tmp_ds['gridlat_0'].values - lat)**2), 
                                                     tmp_ds['gridlon_0'].shape)
                n = n + 1

        if debug:
            print('iind =', iind)
            print('jind =', jind)
 
        i_llcrnr = np.amin(iind)
        i_urcrnr = np.amax(iind)
        j_llcrnr = np.amin(jind)
        j_urcrnr = np.amax(jind)

        return i_llcrnr, i_urcrnr, j_llcrnr, j_urcrnr


    def _subset_ens_domain(self, min_lon, max_lon, min_lat, max_lat, zind, zfield='lv_HYBL2'):
        """
        Create an ensemble subset for a certain spatial domain

        Parameters
        ----------
        min_lon : float
            Minimum longitude (deg E, use negatives for W. hemisphere)     
        max_lon : float
            Maximum longitude (deg E, use negatives for W. hemisphere)     
        min_lat : float
            Minimum latitude (deg N)     
        max_lat : float
            Maximum latitude (deg N)     
        zind : list of integers
            Vertical index 
        zfield : string, optional
            Field to apply zind to. Set to None to not subset in the vertical.

        Returns
        -------
        subset_ds : xr.DataSet
            Ensemble output subsetted to the desired domain

        """

        crnr_ind = self._get_corner_ind(min_lon, max_lon, min_lat, max_lat)

        subset_ds = {}
        zdict = {zfield:zind}
        for key in self.mem_names:
            if zfield is None:
                subset_ds[key] = self.ds[key].sel(xgrid_0=slice(crnr_ind[2], crnr_ind[3]), 
                                                  ygrid_0=slice(crnr_ind[0], crnr_ind[1]))
            else:
                subset_ds[key] = self.ds[key].sel(xgrid_0=slice(crnr_ind[2], crnr_ind[3]), 
                                                  ygrid_0=slice(crnr_ind[0], crnr_ind[1]), 
                                                  **zdict)

        return subset_ds


    def _create_state_matrix(self, fields, thin=1, 
                             loc_fields=['lv_HYBL2', 'gridlat_0', 'gridlon_0']):
        """
        Create state matrix

        Parameters
        ----------
        fields : list of strings
            Fields to include in state matrix (for now, these must be 3D)
        thin : integer, optional
            Only use every nth point in state matrix
        loc_fields : list of strings, optional
            Fields to include in the "location" entry of the state matrix

        Returns
        -------
        state_matrix: dictionary
            State matrix with three entries: Data, field names, and locations

        Notes
        -----
        The locations in the state matrix come from the first ensemble member. It is asumed that
        every ensemble member has the same grid

        """

        loop_list = [self.subset_ds[key] for key in self.mem_names]
        len_state_vect = np.sum(np.array([loop_list[0][f][:, ::thin, ::thin].size for f in fields]))
        state_matrix = {'data':np.zeros([len_state_vect, len(loop_list)]),
                        'vars':np.concatenate([np.array([f]*loop_list[0][f][:, ::thin, ::thin].size) for f in fields]),
                        'loc':np.zeros([len_state_vect, len(loc_fields)])}
        
        # Populate data fields
        for i, ds in enumerate(loop_list):
            state_matrix['data'][:, i] = np.concatenate([np.ravel(ds[f][:, ::thin, ::thin]) for f in fields])

        # Populate location fields
        sample_ds = self.subset_ds[self.mem_names[0]]
        nz, ny, nx = sample_ds[fields[0]][:, ::thin, ::thin].shape
        for i, f in enumerate(loc_fields):
            loc_shape = len(sample_ds[f].shape)
            if loc_shape == 1:
                loc1d = np.ravel(np.tile(sample_ds[f].values[:, np.newaxis, np.newaxis], (1, ny, nx)))
            elif loc_shape == 2:
                loc1d = np.ravel(np.tile(sample_ds[f].values[np.newaxis, ::thin, ::thin], (nz, 1, 1)))
            elif loc_shape == 3:
                loc1d = np.ravel(sample_ds[f][:, ::thin, ::thin])
            state_matrix['loc'][:, i] = np.array(list(loc1d)*len(fields))
        
        return state_matrix


    def _compute_ens_stats(self, stat_fct={'mean':np.mean, 'std':np.std, 'med':np.median}):
        """
        Compute ensemble statistics for the state matrix
 
        Parameters
        ----------
        stat_fct : dictionary, optional
            Statistics to compute

        Returns
        -------
        stats : dictionary
            Ensemble statistics

        """

        stats = {}
        for key in stat_fct.keys():
            stats[key] = stat_fct[key](self.state_matrix['data'], axis=1)
           
        return stats
    

    def _compute_ens_deviations(self):
        """
        Compute deviations from ensemble mean
 
        Returns
        -------
        ens_dev : array
            Deviations from ensemble mean

        """

        ens_dev = self.state_matrix['data'] - self.state_matrix['mean'][:, np.newaxis]           

        return ens_dev


    def _compute_bec(self):
        """
        Compute background error covariance matrix

        Returns
        -------
        be_cov : array 
            Background error covariance matrix
        be_corr
            Background error correlation matrix

        """

        be_corr = np.corrcoef(self.state_matrix['data'])
        be_cov = np.cov(self.state_matrix['data'])
           
        return be_cov, be_corr
    

    def ens_stats(self, field, stat_fct=np.mean, stat_kw={}):
        """
        Compute ensemble statistics for any field in self.subset_ds
 
        Parameters
        ----------
        field : string
            Field to compute statistics for. Must exist in self.subset_ds
        stat_fct : function
            Function for computing statistics
        stat_kw : dictionary, optional
            Keyword arguments passed to stat_fct

        Returns
        -------
        stat : np.array
            Output of stat_fct applied to the desired field

        """

        # Create ND array, where first dimension is the ensemble members
        n = len(self.mem_names)
        full = np.zeros(tuple([n] + [d for d in self.subset_ds[self.mem_names[0]][field].shape]))
        for i, m in enumerate(self.mem_names):
            full[i] = self.subset_ds[m][field].values

        # Apply statistical function
        stat = stat_fct(full, axis=0, **stat_kw)
           
        return stat


    def check_pts_in_subset_domain(self, points):
        """
        Check to see if points lie within the subset domain

        Parameters
        ----------
        points : array
            2D array of (lon, lat) point to check

        Returns
        -------
        indomain : array
            1D boolean array where values are True is points lie in the domain

        """

        # Extract (lat, lon) coordinates and grid dimensions
        lon2d = self.subset_ds[self.mem_names[0]]['gridlon_0'].values
        lat2d = self.subset_ds[self.mem_names[0]]['gridlat_0'].values
        idim, jdim = lon2d.shape

        # Define points along the edge of the domain
        edge_pts = [[lon2d[i, j], lat2d[i, j]] 
                    for i, j in zip(list(range(idim)) + [idim-1]*jdim + list(range(idim-1, -1, -1)) + [0]*jdim,
                                    [0]*idim + list(range(jdim)) + [jdim-1]*idim + list(range(jdim-1, -1, -1)))]

        # Create path object
        path = mplPath.Path(edge_pts)

        indomain = path.contains_points(points)

        return indomain

   
    def _subset_bufr(self, subset, nonan_field=np.nan, DHR=0):
        """
        Subset BUFR obs to only contin certain ob subsets in a certain domain with a certain field
        is not NaN

        Parameters
        ----------
        subset : list of strings
            List of observation subsets to retain
        nonan_field : string, optional
            Only retain rows if this field is not a NaN. Set to NaN to not use
        DHR : float, optional
            If multiple obs exist for a single SID, only retain the obs closest to DHR. Set to NaN
            to turn off 

        Returns
        -------
        red_ob_csv : pd.DataFrame
            Observation subset

        """

        # Only retain rows with a certain ob subset
        keep_idx = np.zeros(len(self.verif_obs.df))
        for s in subset:
            keep_idx[self.verif_obs.df['subset'] == s] = 1
        red_ob_csv = self.verif_obs.df.loc[keep_idx == 1, :].copy()
        red_ob_csv.reset_index(inplace=True, drop=True)

        # Subset over the same domain as the ensemble
        spatial_idx = np.where((red_ob_csv['YOB'] >= self.lat_limits[0]) &
                               (red_ob_csv['YOB'] <= self.lat_limits[1]) &
                               (red_ob_csv['XOB'] >= (360 + self.lon_limits[0])) &
                               (red_ob_csv['XOB'] <= (360 + self.lon_limits[1])))[0]
        red_ob_csv = red_ob_csv.iloc[spatial_idx, :]
        red_ob_csv.reset_index(inplace=True, drop=True)
    
        # Remove any additional points that might life outside the domain
        ob_pts = [[x, y] for x, y in zip(red_ob_csv['XOB'] - 360., red_ob_csv['YOB'])]
        spatial_idx2 = self.check_pts_in_subset_domain(ob_pts)
        red_ob_csv = red_ob_csv.iloc[spatial_idx2, :]
        red_ob_csv.reset_index(inplace=True, drop=True)
 
        # Only retain rows where the nonan_field is not a NaN
        if nonan_field == nonan_field:
            red_ob_csv = red_ob_csv.loc[~np.isnan(red_ob_csv[nonan_field])]
            red_ob_csv.reset_index(inplace=True, drop=True)

        # Only retain the single ob closest to DHR for each site
        if ~np.isnan(DHR):
            all_sid = np.unique(red_ob_csv['SID'])
            keep_idx = [np.abs(red_ob_csv['DHR'].loc[red_ob_csv['SID'] == s] - DHR).idxmin() for s in all_sid]
            red_ob_csv = red_ob_csv.loc[keep_idx]
            red_ob_csv.reset_index(inplace=True, drop=True)

        return red_ob_csv 
    

    def preprocess_model_ceil(self, kwargs={'no_ceil':np.nan}):
        """
        Convert model ceiling heights to AGL and set missing values to NaN

        Parameters
        ----------
        ens_obj : pyDA_utils.ensemble_utils.ensemble
            Ensemble object
        kwargs : Dictionary, optional
            Keyword arguments passed to uppp.compute_ceil_agl()

        Returns
        -------
        ens_obj : pyDA_utils.ensemble_utils.ensemble
            Ensemble object
            
        """

        for m in self.mem_names:
            print(f'computing ceiling AGL heights for {m}')
            self.subset_ds[m] = uppp.compute_ceil_agl(self.subset_ds[m], **kwargs)


    def interp_model_2d(self, field, lat, lon, zind=np.nan, method='nearest', verbose=False):
        """
        Interpolate model output to certain locations

        Parameters
        ----------
        field : string
            Field to interpolate
        lat : array
            Latitudes to interpolate to (deg N)
        lon : array
            Longitudes to interpolate to (deg E in range -180 to 180)
        zind : integer, optional
            Vertical index to use if field is 3D. Set to NaN for 2D fields
        method : string, optional
            Interpolation method. Only options are 'nearest' and 'linear' at the moment
        verbose: boolean, optional
            Option for verbose output

        Returns
        -------
        interp_ens_df : DataFrame
            DataFrame holding interpolated output from each ensemble member

        """

        # Obtain lat/lon arrays, which are the same for each ensemble member
        mlat1d = np.ravel(self.subset_ds[self.mem_names[0]]['gridlat_0'])
        mlon1d = np.ravel(self.subset_ds[self.mem_names[0]]['gridlon_0'])

        interp_ens_dict = {'XOB':lon, 'YOB':lat}
        for mem in self.mem_names:
            if verbose:
                print('Performing interpolation for {n}'.format(n=mem))
            if np.isnan(zind):
                z = np.ravel(self.subset_ds[mem][field].values)
            else:
                z = np.ravel(self.subset_ds[mem][field][zind, :, :].values)
            if method == 'nearest':
                interp_fct = si.NearestNDInterpolator(list(zip(mlon1d, mlat1d)), z)
            elif method == 'linear':
                interp_fct = si.LinearNDInterpolator(list(zip(mlon1d, mlat1d)), z)
            else:
                print("Only 'nearest' and 'linear' interpolation are currently supported")        
            interp_ens_dict[mem] = interp_fct(lon, lat)

        interp_ens_df = pd.DataFrame(interp_ens_dict)

        return interp_ens_df
    

    def postage_stamp_contourf(self, field, nrows, ncols, klvl=np.nan, figsize=(10, 10), title='',
                               plt_kw={}, verbose=False):
        """
        Make filled contour postage stamp plot (i.e., each ensemble member plotted individually)

        Parameters
        ----------
        field : string
            UPP field to plot
        nrows : integer
            Number of rows
        ncols : integer
            Number of columns
        klvl : integer, optional
            Vertical level to plot the field at. Set to NaN for 2D fields
        figsize : tuple of floats, optional
            Figure size
        title : string, optional
            Title
        plt_kw : dictionary, optional
            Other keyword arguments passed to pmd.contourf
        verbose : boolean, optional
            Option to print verbose output

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure with the postage stamp plot

        """

        fig = plt.figure(figsize=figsize)
        for i, key in enumerate(self.mem_names):
            if verbose:
                print('plotting {mem}'.format(mem=key))
            plot_obj = pmd.PlotOutput([self.subset_ds[key]], 'upp', fig, nrows, ncols, i+1)

            # Skip plotting if < 2 NaN
            if np.isnan(klvl):
                make_plot = np.sum(~np.isnan(self.subset_ds[key][field])) > 1
            else:
                make_plot = np.sum(~np.isnan(self.subset_ds[key][field][klvl, :, :])) > 1
            if make_plot:
                plot_obj.contourf(field, cbar=False, **plt_kw)
                cax = plot_obj.cax
                meta = plot_obj.metadata['contourf0']
            else:
                if verbose:
                    print('skipping plot for {mem}'.format(mem=key))
                plot_obj.ax = fig.add_subplot(nrows, ncols, i+1, projection=plot_obj.proj)

            plot_obj.config_ax(grid=False)
            plot_obj.set_lim(self.lat_limits[0], self.lat_limits[1], 
                             self.lon_limits[0], self.lon_limits[1])
            plot_obj.ax.set_title(key, size=14)
    
        cb_ax = fig.add_axes([0.915, 0.02, 0.02, 0.9])
        cbar = plt.colorbar(cax, cax=cb_ax, orientation='vertical', aspect=35)
        cbar.set_label('%s%s (%s)' % (meta['interp'], 
                                      meta['name'], 
                                      meta['units']), size=14)
        if ~np.isnan(klvl):
            tmp_ds = self.subset_ds[self.mem_names[0]]
            title = '{t} (avg z = {z:.1f} m)'.format(t=title, z=float(np.mean(tmp_ds['HGT_P0_L105_GLC0'][klvl, :, :] -
                                                                              tmp_ds['HGT_P0_L1_GLC0'])))
        plt.suptitle(title, size=18)

        return fig

    
    def plot_bufr_obs(self, axes, field, bufr_subset, nonan_field=np.nan, scatter_kw={}):
        """
        Plot BUFR obs on a pre-existing set of axes

        Parameters
        ----------
        axes : list of cartopy.mpl.geoaxes.GeoAxesSubplot objects
            Axes to plot BUFR obs on
        field : string
            BUFR field to plot
        bufr_subset : list of strings
            List of ob subsets to retain
        nonan_field : string, optional
            Only retain BUFR obs if this field is not a NaN
        scatter_kw : dictionary, optional
            Keyword arguments passed to the scatter function
        
        """
 
        # Subset the BUFR obs
        plot_csv = self._subset_bufr(bufr_subset, nonan_field=nonan_field)

        # Plot BUFR obs
        for ax in axes:
            ax.scatter(plot_csv['XOB'], plot_csv['YOB'], c=plot_csv[field], 
                       transform=ccrs.PlateCarree(), **scatter_kw)


    def plot_state_vector(self, entry, field, fig, nrows, ncols, axnum, zind=np.nan, mem_name=None,
                          bec_idx=np.nan, plot_cbar=True, plot_zavg=True, ctr_cmap_0=False, 
                          pcm_kw={}, cbar_kw={}, debug=False):
        """
        Plot a vector from the state_matrix

        Parameters
        ----------
        entry : string
            Entry in state_matrix to plot
        field : string
            Meteorological field to plot
        fig : matplotlit.pyplot.figure
            Figure to add subplot to
        nrows : integer
            Number of rows for subplots on the figure
        ncols : integer
            Number of columns for subplots on the figure
        axnum : integer
            Axes number to plot
        zind : integer, optional
            Vertical level to plot. Set to NaN for 2D fields
        mem_name : string, optional
            Ensemble member to plot
        bec_idx : integer, optional
            Index corresponding to the "target" index for plotting BECs
        plot_cbar : boolean, optional
            Option to plot colorbar
        plot_zavg : boolean, optional
            Option to plot average height
        ctr_cmap_0 : boolean, optional
            Option to center the colorbar on 0
        pcm_kw : dictionary, optional
            Keyword arguments used in pcolormesh
        cbar_kw : dictionary, optional
            Keyword arguments used when creating the colorbar
        debug : boolean, optional
            Additional output for debugging

        Returns
        -------
        ax : matplotlib.axes
            Axes plot was added to

        """

        # Unravel state_matrix
        sample_ds = self.subset_ds[self.mem_names[0]]
        if mem_name != None:
            mem_idx = np.where(self.mem_names == mem_name)[0][0]
            tmp_vect = np.squeeze(self.state_vector[entry][:, mem_idx])
        elif ~np.isnan(bec_idx):
            tmp_vect = np.squeeze(self.state_matrix[entry][bec_idx, :])
        else:
            tmp_vect = self.state_matrix[entry]

        if debug:
            print()
            print('shape of tmp_vect =', np.shape(tmp_vect))
            print('shape of tmp_vect[state_matrix[vars] == field] =', 
                  np.shape(tmp_vect[self.state_matrix['vars'] == field]))
            print('shape of sample_ds[field] =', np.shape(sample_ds[field]))

        plot_array = np.reshape(tmp_vect[self.state_matrix['vars'] == field], sample_ds[field].shape)
        if ~np.isnan(zind):
            plot_array = plot_array[zind, :, :]

        # Specifiy vmin and vmax if ctr_cmap_0 = True
        if ctr_cmap_0:
            max_abs = np.amax(np.abs(np.array([np.percentile(plot_array, 99), 
                                               np.percentile(plot_array, 1)])))
            pcm_kw['vmin'] = -max_abs
            pcm_kw['vmax'] = max_abs

        # Make plot
        ax = fig.add_subplot(nrows, ncols, axnum, projection=ccrs.LambertConformal())
    
        cax = ax.pcolormesh(sample_ds['gridlon_0'], sample_ds['gridlat_0'], plot_array, 
                            transform=ccrs.PlateCarree(), **pcm_kw)
    
        ax.coastlines('50m')
        borders = cfeature.NaturalEarthFeature(category='cultural',
                                               scale='50m',
                                               facecolor='none',
                                               name='admin_1_states_provinces')
        ax.add_feature(borders, lw=1, edgecolor='gray')
        ax.set_extent([self.lon_limits[0], self.lon_limits[1], self.lat_limits[0], self.lat_limits[1]])
    
        # Add colorbar
        if plot_cbar:
            cbar = plt.colorbar(cax, ax=ax, **cbar_kw)
            cbar.set_label('{entry} {field}'.format(entry=entry, field=field), size=14)
         
        # Plot average height in axes title
        if plot_zavg and ~np.isnan(zind):
            title = 'avg z = {z:.1f} m'.format(z=float(np.mean(sample_ds['HGT_P0_L105_GLC0'][zind, :, :] -
                                                               sample_ds['HGT_P0_L1_GLC0'])))
            ax.set_title(title, size=14)

        return ax
       

    def plot_ens_dev_hist(self, field, ax, hist_kw={}):
        """
        Plot histograms of deviations from the ensemble mean

        Parameters
        ----------
        field : string
            UPP field to plot
        ax : matplotlib.axes
            Axes to add plot to
        hist_kw : dictionary, optional
            Keyword arguments passed to matplotlib.pyplot.hist
        
        Returns
        -------
        ax : matplotlib.axes
            Axes with histograms of deviations plotted

        """

        idx = np.where(self.state_matrix['vars'] == field)
        ax.hist(self.state_matrix['ens_dev'][idx].flatten(), **hist_kw)
        ax.set_title(field, size=18)
        ax.set_xlabel(self.subset_ds[self.mem_names[0]][field].attrs['units'], size=14)

        return ax
    

    def plot_skewts(self, lon, lat, fig, nrows=1, ncols=1, nplot=1, names=[], skew_kw={}):
        """
        Plot Skew-T, logp diagrams for a series of ensemble members

        Parameters
        ----------
        lon : float
            Longitude to plot skew-T diagram for (deg E)
        lat : float
            Latitude to plot skew-T diagram for (deg N)
        fig : matplotlib.pyplot.figure
            Figure to add skew-T to
        nrows : integer, optional
            Number of subplot rows in the figure
        ncols : integer, optional
            Number of subplot columns in the figure
        nplot : integer, optional
            Subplot number used to plot the skew-T
        names : list, optional
            Ensemble members to plot. If blank, plot all members
        skew_kw : dictionary, optional
            Keyword arguments passed to the skewt function

        Returns
        -------
        plot_obj : pmd.PlotOutput object
            Object containing the skew-T plot

        """

        # Use all ensemble members if names is an empty list
        if len(names) == 0:
            names = self.mem_names
        
        # Make plots
        for i, mem in enumerate(names):
            plot_obj = pmd.PlotOutput([self.subset_ds[mem]], 'upp', fig, nrows, ncols, nplot)
            plot_obj.skewt(lon, lat, **skew_kw)
            if (len(names) > 1) and ('skew' not in skew_kw.keys()):
                skew_kw['skew'] = plot_obj.skew

        return plot_obj



    def save_subset_ens(self, fname):
        """
        Save the subset ensemble datasets to a netCDF file

        Parameters
        ----------
        fname : string
            NetCDF file to save data to
        
        Returns
        -------
        None

        """ 

        # Combine all ensemble data into a single dataset
        for i, mem in enumerate(self.mem_names):
            self.subset_ds[mem].expand_dims(dim='num', axis=0)
            self.subset_ds[mem].assign_coords(num=(i+1))
        concat_ds = xr.concat([self.subset_ds[m] for m in self.mem_names], 'num')
        concat_ds['names'] = xr.DataArray(self.mem_names, dims=('num'))
        concat_ds.attrs['lat_limits'] = self.lat_limits
        concat_ds.attrs['lon_limits'] = self.lon_limits

        concat_ds.to_netcdf(fname)


def read_subset_ens_nc(fname):
    """
    Read ensemble subset data from a netCDF file

    Parameters
    ----------
    fname : string
        NetCDF file containing subset ensemble data

    Returns
    -------
    ens_obj: eu.ensemble object
        Ensemble data

    """

    # Read in netCDF file contents
    concat_ds = xr.open_dataset(fname)

    ens_obj = ensemble({})
    ens_obj.subset_ds = {}
    for i, mem in enumerate(concat_ds['names'].values):
        ens_obj.subset_ds[mem] = copy.deepcopy(concat_ds.sel(num=i))
    ens_obj.mem_names = copy.deepcopy(concat_ds['names'].values)
    ens_obj.lat_limits = copy.deepcopy(concat_ds.attrs['lat_limits'])
    ens_obj.lon_limits = copy.deepcopy(concat_ds.attrs['lon_limits'])

    return ens_obj 


"""
End ensemble_utils.py
"""

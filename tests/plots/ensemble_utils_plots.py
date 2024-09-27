"""
Example Plots for ensemble_utils.py

Inputs
------
sys.argv[1] : YAML file with inputs

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import sys

import pyDA_utils.ensemble_utils as eu


#---------------------------------------------------------------------------------------------------
# Plots
#---------------------------------------------------------------------------------------------------

# Read in YAML input
fname = sys.argv[1]
with open(fname, 'r') as fptr:
    param = yaml.safe_load(fptr)

# Create ensemble object
str_format = param['str_format']
prslev_fnames = {}
natlev_fnames = {}
for i in range(1, param['nmem']+1):
    prslev_fnames['mem{num:04d}'.format(num=i)] = str_format.format(num=i, lev='prslev')
    natlev_fnames['mem{num:04d}'.format(num=i)] = str_format.format(num=i, lev='natlev')

ens_obj = eu.ensemble(natlev_fnames, extra_fnames=prslev_fnames, 
                      extra_fields=param['prslev_vars'], 
                      bufr_csv_fname=param['bufr_fname'], 
                      lat_limits=[param['min_lat'], param['max_lat']],
                      lon_limits=[param['min_lon'], param['max_lon']],
                      zind=param['z_ind'],
                      state_fields=param['state_vars'],
                      bec=False)

# Skew-T, logp diagram
if param['skewt']:
    lat = 0.5*(ens_obj.lat_limits[0] + ens_obj.lat_limits[1])
    lon = 0.5*(ens_obj.lon_limits[0] + ens_obj.lon_limits[1])
    fig = plt.figure(figsize=(8, 8))
    skew_kw = {'hodo':False, 
               'barbs':False,
               'Tplot_kw':{'linewidth':1, 'color':'r'},
               'TDplot_kw':{'linewidth':1, 'color':'b'}}
    plot_obj = ens_obj.plot_skewts(lon, lat, fig, skew_kw=skew_kw)
    plot_obj.skew.ax.set_xlim(-20, 20)
    plot_obj.skew.ax.set_ylim(1000, 500)
    plt.savefig('ens_skewt.png')


"""
End test_ensemble_utils.py
"""

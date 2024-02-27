
import pyDA_utils.superob_prepbufr as sp
import pyDA_utils.map_proj as mp
import metpy.interpolate as mi
import numpy as np

fname = './tests/data/202204291500.rap.fake.prepbufr.for_superob_test.csv'
sample_pb = sp.superobPB(fname)
sample_pb.map_proj = mp.ll_to_xy_lc
sample_pb.map_proj_kw = {'dx':3, 'knowni':899, 'knownj':529}

# Create input df
grid_fname='./tests/data/RRFS_grid_max.nc'
sample_pb.assign_superob('grid', grouping_kw={'grid_fname':grid_fname})
qc_df = sample_pb.qc_obs(field='TQM', thres=2)

# Obtain superob coordinates
superobs_in = sample_pb.reduction_superob(var_dict={})

superobs_no_metpy = sample_pb.reduction_hor_cressman(qc_df, superobs_in, 'TOB', R=1, use_metpy=False)
superobs_metpy = sample_pb.reduction_hor_cressman(qc_df, superobs_in, 'TOB', R=1, use_metpy=True)

# First superob group
g = superobs_in['superob_groups'].values[0]
subset_df = qc_df.loc[qc_df['superob_groups'] == g].copy()
ob_pts = np.array([[x, y] for x, y in zip(subset_df['XMP'], subset_df['YMP'])])
ob_vals = subset_df['TOB'].values
superob_pt = np.array([[superobs_in['XMP'].values[0], superobs_in['YMP'].values[0]]])
print('No MetPy =', superobs_no_metpy[0])
print('With MetPy =', superobs_metpy[0])
print('Using Mean =', np.mean(subset_df['TOB']))

metpy2 = mi.interpolate_to_points(ob_pts, ob_vals, superob_pt, interp_type='cressman', search_radius=1)
print('With MetPy (offline) =', metpy2)

"""
Create Sample PrepBUFR Data for bufr Tests

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pyDA_utils.bufr as bufr
import numpy as np


#---------------------------------------------------------------------------------------------------
# Input Parameters
#---------------------------------------------------------------------------------------------------

in_fname = '/mnt/lfs4/BMC/wrfruc/murdzek/real_obs/decoded_bufr_for_cloudDA/202202011200.rap.prepbufr.csv'
out_fname = './202202011200.rap.prepbufr.for_bufr_test.csv'


#---------------------------------------------------------------------------------------------------
# Only Save Certain Obs
#---------------------------------------------------------------------------------------------------

in_bufr = bufr.bufrCSV(in_fname)

# Keep all type 187 obs
keep_idx = np.where(np.isclose(in_bufr.df['TYP'], 187))[0]

# Keep first 20 SID for other obs
all_types = np.unique(in_bufr.df['TYP'])
for t in all_types:
    if ~np.isclose(t, 187):
        keep_idx = np.concatenate([keep_idx, np.where(np.isclose(in_bufr.df['TYP'], t))[0][:20]])

out_df = in_bufr.df.iloc[keep_idx, :].copy()

bufr.df_to_csv(out_df, out_fname)


"""
End create_PB_for_bufr_test.py
"""

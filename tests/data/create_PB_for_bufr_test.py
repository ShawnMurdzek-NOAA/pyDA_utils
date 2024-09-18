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

in_fname = '/mnt/lfs5/BMC/wrfruc/murdzek/real_obs/decoded_bufr_for_cloudDA/202202011200.rap.prepbufr.csv'
out_fname = './202202011200.rap.prepbufr.for_bufr_test.csv'


#---------------------------------------------------------------------------------------------------
# Only Save Certain Obs
#---------------------------------------------------------------------------------------------------

in_bufr = bufr.bufrCSV(in_fname)

# Keep all type 187 obs (never SYNOP according to PB table 5)
keep_idx = np.where(np.isclose(in_bufr.df['TYP'], 187))[0]

# Keep first 20 SID for other unrestricted obs
# NOTE: Various mesonet and SYNOP obs are restricted. Aircraft obs are only restricted <48 hrs 
# after collection
keep_types = [120, 126, 130, 131, 133, 134, 135, 153, 
              220, 224, 227, 230, 231, 233, 234, 235, 242, 243, 250, 252, 253, 254]
all_types = np.unique(in_bufr.df['TYP'].values)
for t in all_types:
    if t in keep_types:
        keep_idx = np.concatenate([keep_idx, np.where(np.isclose(in_bufr.df['TYP'], t))[0][:20]])

out_df = in_bufr.df.iloc[keep_idx, :].copy()

bufr.df_to_csv(out_df, out_fname)


"""
End create_PB_for_bufr_test.py
"""

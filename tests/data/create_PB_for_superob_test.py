"""
Create Sample PrepBUFR Data for superob_prepbufr Tests

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

in_fname = '/work2/noaa/wrfruc/murdzek/nature_run_spring/obs/uas_hspace_150km_ctrl/perf_uas_csv/202204291500.rap.fake.prepbufr.csv'
out_fname = './202204291500.rap.fake.prepbufr.for_superob_test.csv'


#---------------------------------------------------------------------------------------------------
# Edit Test Data
#---------------------------------------------------------------------------------------------------

in_bufr = bufr.bufrCSV(in_fname)

out_df = in_bufr.df.loc[(in_bufr.df['SID'] == "'UA000001'") | 
                        (in_bufr.df['SID'] == "'UA000002'") |
                        (in_bufr.df['SID'] == "'UA000003'")]

# Force one of the UAS to be very close to another UAS
x1 = out_df.loc[out_df['SID'] == "'UA000001'", 'XOB'].values[0]
y1 = out_df.loc[out_df['SID'] == "'UA000001'", 'YOB'].values[0]
out_df.loc[out_df['SID'] == "'UA000002'", 'XOB'] = x1 + 0.005
out_df.loc[out_df['SID'] == "'UA000002'", 'YOB'] = y1 + 0.005

bufr.df_to_csv(out_df, out_fname)


"""
End create_PB_for_superob_test.py
"""

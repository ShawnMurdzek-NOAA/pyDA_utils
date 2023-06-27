"""
Script to Create BUFR Precision CSV

shawn.s.murdzek@noaa.gov
Date Created: 14 April 2022
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np


#---------------------------------------------------------------------------------------------------
# Create CSV File
#---------------------------------------------------------------------------------------------------

csv_fname = 'bufr_precision.csv'

ob_typ = np.arange(100, 300)
prec_dict = {}
prec_dict['TYP'] = ob_typ
ntyp = len(ob_typ)

# Number of decimal points for each ob
prec_dict['TOB_ndec'] = [1] * ntyp
prec_dict['QOB_ndec'] = [0] * ntyp
prec_dict['POB_ndec'] = [1] * ntyp
prec_dict['UOB_ndec'] = [1] * ntyp
prec_dict['VOB_ndec'] = [1] * ntyp
prec_dict['WSPD_ndec'] = [1] * ntyp  # In kts!
prec_dict['WDIR_ndec'] = [0] * ntyp
prec_dict['ELV_ndec'] = [0] * ntyp
prec_dict['ZOB_ndec'] = [0] * ntyp
prec_dict['PWO_ndec'] = [1] * ntyp

# Minimum value for each ob (values below this value are set to 0)
# NaN = no minimum value
prec_dict['TOB_min'] = [np.nan] * ntyp
prec_dict['QOB_min'] = [0] * ntyp
prec_dict['POB_min'] = [0] * ntyp
prec_dict['UOB_min'] = [np.nan] * ntyp
prec_dict['VOB_min'] = [np.nan] * ntyp
prec_dict['WSPD_min'] = [0] * ntyp  # In kts!
prec_dict['WDIR_min'] = [0] * ntyp
prec_dict['ELV_min'] = [0] * ntyp
prec_dict['ZOB_min'] = [np.nan] * ntyp
prec_dict['PWO_min'] = [0] * ntyp

prec_df = pd.DataFrame(prec_dict)
prec_df.set_index('TYP', inplace=True)

# Change settings for specific ob types
prec_df.loc[130, 'TOB_ndec'] = 0
prec_df.loc[187, 'TOB_ndec'] = 0
prec_df.loc[220, 'WSPD_ndec'] = 0
prec_df.loc[230, 'WSPD_ndec'] = 0
prec_df.loc[231, 'WSPD_ndec'] = 0
prec_df.loc[233, 'WSPD_ndec'] = 0
prec_df.loc[234, 'WSPD_ndec'] = 0
prec_df.loc[235, 'WSPD_ndec'] = 0
prec_df.loc[284, 'WSPD_ndec'] = 0
prec_df.loc[287, 'WSPD_ndec'] = 0
prec_df.loc[293, 'WSPD_ndec'] = 0

# Create CSV
prec_df.to_csv(csv_fname)


"""
create_bufr_precision_csv.py
"""


import numpy as np
import datetime as dt
import xarray as xr

import enkf

# Read in sample ensemble
N = 3
x_b = []
Hx_b = []
for i in range(N):
    ds = xr.open_dataset(f'../../direct_ceilometer_DA/tests/sample_data/mpas/mem00{i+1}/mpasout.2024-05-27_04.00.00.TEST.nc')
    Hx_b.append(ds['theta'].values[0, 300, 2])
    x_b.append(np.ravel(ds['theta'].values))

x_b = np.array(x_b).T
Hx_b = np.array(Hx_b)
y_0 = np.mean(Hx_b) + 2
ob_var = 1

n = 50
time = np.zeros(n)
for i in range(n):
    start = dt.datetime.now()
    enkf_obj = enkf.enkf_1ob(x_b, y_0, Hx_b, ob_var, jit=False)
    enkf_obj.EnSRF()
    time[i] = (dt.datetime.now() - start).total_seconds()
print(f"Avg elapsed time = {np.mean(time)} s")

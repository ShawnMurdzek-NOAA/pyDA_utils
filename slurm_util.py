"""
Functions Related to Using SLURM on HPC

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import os

#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def job_info(job_df, user, maxtries):
    """
    Returns information about active and completed jobs from Slurm

    Parameters 
    ----------
    job_df : pd.DataFrame
        DataFrame containing job information
    user : string
        User ID
    maxtries : integer
        Maximum number of attempts for each job

    Returns
    -------
    job_df : pd.DataFrame
        DataFrame containing updated job information
    njobs : integer
        Number jobs queued or running 

    """

    # Determine number of queued/running jobs
    njobs = len(os.popen('squeue -u %s' % user).read().split('\n')) - 2

    # Determine job statuses and mark jobs as either completed or need to be resubmitted
    idx = np.where(np.logical_and(job_df['submitted'], ~job_df['completed']))[0]
    for i in idx:
        jobID = job_df.loc[i, 'jobID']
        sacct_out = os.popen('sacct --jobs=%d' % jobID).read().split('\n')[2].split()
        if sacct_out[5] == 'COMPLETED':
            if sacct_out[6] == '0:0':
                job_df.loc[i, 'completed'] = True
            else:
                if job_df.loc[i, 'tries'] < maxtries:
                    job_df.loc[i, 'submitted'] = False

    return job_df, njobs


"""
End slurm_util.py
"""

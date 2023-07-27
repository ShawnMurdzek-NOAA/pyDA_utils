"""
Functions Related to Using SLURM on HPC

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd


#---------------------------------------------------------------------------------------------------
# Job List Class
#---------------------------------------------------------------------------------------------------

class job_list:
    
    def __init__(self, fname=None, df=None, jobs=[]):
        if fname != None:
            self.df = pd.read_csv(fname)
        elif df != None:
            self.df = df
        else:
            nrow = len(jobs)
            tmp = {'file':jobs, 'submitted':[False]*nrow, 'completed':[False]*nrow, 
                   'jobID':np.zeros(nrow, dtype=int)*np.nan, 'tries':np.zeros(nrow, dtype=int)}
            self.df = pd.DataFrame(tmp)


    def save(self, save_fname):
        self.df.to_csv(save_fname, index=False)
        return None


    def update(self, user, maxtries):
        """
        Updates job_list DataFrame about active and completed jobs from Slurm

        Parameters 
        ----------
        user : string
            User ID
        maxtries : integer
            Maximum number of attempts for each job

        """

        idx = np.where(np.logical_and(self.df['submitted'], ~self.df['completed']))[0]
        for i in idx:
            jobID = self.df.loc[i, 'jobID']
            sacct_out = os.popen('sacct --jobs=%d' % jobID).read().split('\n')[2].split()
            if sacct_out[5] == 'COMPLETED':
                if sacct_out[6] == '0:0':
                    self.df.loc[i, 'completed'] = True
                else:
                    if self.df.loc[i, 'tries'] < maxtries:
                        self.df.loc[i, 'submitted'] = False

        return None


    def submit_jobs(self, user, max_jobs, verbose=True):
        """
        Submit jobs until all jobs are submitted or the max number of active jobs is reached

        Parameters
        ----------
        user : string
            User ID
        max_jobs : integer
            Maximum number of active jobs
        verbose : boolean, optional
            Print the name of each of the jobs being submitted

        """

        self.df.reset_index(drop=True, inplace=True)
        idx = np.where(~self.df['submitted'])[0]
        njobs = job_number(user)
        njobs_submit = max(min(max_jobs - njobs, len(idx)), 0)
        if verbose:
            print('initial njobs = %d' % njobs)
            print('max allowed njobs = %d' % max_jobs)
            print('number potential jobs = %d' % len(idx))
            print('will submit %d jobs' % njobs_submit)
            print()

        for i in idx[:njobs_submit]:
            self.df.loc[i, 'jobID'] = int(os.popen('sbatch %s' % self.df.loc[i, 'file']).read().strip().split(' ')[-1])
            self.df.loc[i, 'tries'] = self.df.loc[i, 'tries'] + 1
            self.df.loc[i, 'submitted'] = True
            if verbose:
                print('submitted job = %d' % self.df.loc[i, 'jobID'])

        return None        


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def job_number(user):
    """
    Returns the number of active jobs

    Parameters 
    ----------
    user : string
        User ID

    Returns
    -------
    njobs : integer
        Number jobs queued or running 

    """

    njobs = len(os.popen('squeue -u %s' % user).read().split('\n')) - 2

    return njobs


"""
End slurm_util.py
"""

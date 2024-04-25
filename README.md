
# Data Assimilation-Related Python Packages

Shawn Murdzek  
NOAA/OAR/Global Systems Laboratory  
shawn.s.murdzek@noaa.gov

[![Lint and Pytest](https://github.com/ShawnMurdzek-NOAA/pyDA_utils/actions/workflows/python-lint-pytest.yml/badge.svg?branch=main)](https://github.com/ShawnMurdzek-NOAA/pyDA_utils/actions/workflows/python-lint-pytest.yml)

## Description

This repo contains several utilities that are helpful in a variety of data assimilation (DA) scenarios, including analyzing output from Gridpoint Statistical Interpolation (GSI), creating synthetic observations for a Observing System Simulation Experiment (OSSE), and plotting model output.

## Testing

I'm slowly adding tests for this project using pytest. Some of these tests are run automatically after a push or PR to the remote. To run these tests manually (recommended before pushing), follow the following steps:  
  
```
cd tests/
pytest
```

To only run the tests in a specific file, run `pytest <filname>`.  
  
To only run a specific test in a specific file, run `pytest <filename>::<testname>`.  
  
To see how long a test takes, use the `--duration=0` flag.

To print stdout from a test, add the `-s` flag. Note that stdout for a test is automatically printed when a test fails.

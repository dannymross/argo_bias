#!/bin/bash
module load r/4.5.0
# ncdf4 links against the netCDF C library at install time; nc-config is not on
# the default PATH, so this module load is required, not optional.
module load netcdf/4.9.2
export R_LIBS="$HOME/R/x86_64-pc-linux-gnu-library/4.5"
Rscript renv_setup.R

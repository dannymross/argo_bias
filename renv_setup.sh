#!/bin/bash
module load r/4.5.0
export R_LIBS="$HOME/R/x86_64-pc-linux-gnu-library/4.5"
Rscript renv_setup.R

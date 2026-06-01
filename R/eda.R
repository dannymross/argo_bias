root <- getwd()
library(tidyverse)
library(data.table)
library(pbapply)
library(geosphere)
library(GpGp)

load(paste0(root, "/data/argo_velo_data_january.RData"))

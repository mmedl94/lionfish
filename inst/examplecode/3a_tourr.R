# source("inst/examplecode/1_load.R")
# source("inst/examplecode/2_preprocess.R")
# Check directly with tourr
animate_xy(d_sub_cl[,1:10], col=d_sub_cl$cl)
animate_xy(d_sub_cl[,1:10], guided_tour(lda_pp(d_sub_cl$cl)), col=d_sub_cl$cl)


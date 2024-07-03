# source("inst/examplecode/1_load.R")
# source("inst/examplecode/2_preprocess.R")
gt2d <- save_history(d_sub,
  tour_path = guided_tour(holes()))
gt2di <- interpolate(gt2d)

gt1di <- gt2di[,1,] #save_history(winterActiv,
#                       tour_path = grand_tour(d=1))
#gt1d <- interpolate(tour_history_1d)

half_range <- max(sqrt(rowSums(d_sub^2)))
col_names <- colnames(d_sub)

# interactive_tour needs data as a matrix
d_sub <- as.matrix(d_sub)

obj1 <- list(type = "2d_tour", obj = gt2di)
obj2 <- list(type = "1d_tour", obj = gt1di)
obj3 <- list(type = "scatter", obj = c("alpineskiing", "excursions"))
obj4 <- list(type = "cat_clust_interface", obj = c("1","2"))

interactive_tour(d_sub,
                 col_names,
                 list(obj1,obj2,obj3,obj4),
                 half_range,
                 preselection = clusters@second,
                 n_max_cols = 2,
                 n_subsets = 10)

library(readr)
library(tourr)
winterActiv <- read_csv("data-raw/winterActiv.csv")
winterActiv <- winterActiv[,1:6]

library(flexclust)
set.seed(1234)
clusters = stepcclust(winterActiv, k=6, nrep=20)

guided_tour_history <- save_history(winterActiv,
                                    tour_path = guided_tour(holes()))
guided_tour_history <- interpolate(guided_tour_history)

tour_history_1d <- save_history(winterActiv,
                                tour_path = grand_tour(d=1))
tour_history_1d <- interpolate(tour_history_1d)

half_range <- max(sqrt(rowSums(winterActiv^2)))
col_names <- colnames(winterActiv)

obj1 <- list(type = "2d_tour", obj = guided_tour_history)
obj2 <- list(type = "1d_tour", obj = tour_history_1d)
obj3 <- list(type = "scatter", obj = c("alpine.skiing", "snowboarding"))
obj4 <- list(type = "cat_clust_interface", obj = c("1","2"))

interactive_tour(winterActiv,
                 col_names,
                 list(obj1,obj2,obj3,obj4),
                 half_range,
                 preselection = clusters@second,
                 n_max_cols = 2,
                 n_subsets = 10)

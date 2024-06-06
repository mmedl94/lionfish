library(tourr)
library(devtools)
library(reticulate)
library(pytour)
f <- apply(flea[,1:6], 2, function(x) (x-mean(x))/sd(x))
guided_tour_history <- save_history(f, 
                                    tour_path = guided_tour(holes()))
grand_tour_history <- save_history(f, 
                                   tour_path = grand_tour())
tour_history_1d <- save_history(f, 
                                tour_path = grand_tour(d=1))

half_range <- max(sqrt(rowSums(f^2)))
col_names <- colnames(f)

init_env()

obj1 <- list(type = "2d_tour", obj = guided_tour_history)
obj2 <- list(type = "1d_tour", obj = tour_history_1d)
obj3 <- list(type = "scatter", obj = c("tars1", "tars2"))
obj4 <- list(type = "hist", obj = "head")

interactive_tour(f,
                 col_names,
                 list(obj1,obj2,obj3,obj4),
                 half_range,
                 n_max_cols=2)

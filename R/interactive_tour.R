#' R wrapper for the interactive_tour function written on python
#'
#' @param data the dataset you want to investigate
#' @param col_names the column names of your dataset
#' @param plot_objects a named list of objects you want to be displayed. Each entry requires a definition of
#' the type of display and a specification of what should be plotted. For tours, a history object, and
#' for histograms and scatterplots, the names of the variables to be displayed have to be provided (see example).
#' @param half_range can be specified directly or be calculated done in the tourr package.
#' @param n_max_cols specifies the maximal number of columns of the grid of the final display.
#'
#' @return -
#' @export
#'
#' @examples
#'library(tourr)
#'library(devtools)
#'library(reticulate)
#'library(pytourr)
#'f <- apply(flea[,1:6], 2, function(x) (x-mean(x))/sd(x))
#'guided_tour_history <- save_history(f,
#'                                    tour_path = guided_tour(holes()))
#'grand_tour_history <- save_history(f,
#'                                   tour_path = grand_tour())
#'tour_history_1d <- save_history(f,
#'                                tour_path = grand_tour(d=1))
#'
#'half_range <- max(sqrt(rowSums(f^2)))
#'col_names <- colnames(f)
#'
#'init_env()
#'
#'obj1 <- list(type = "2d_tour", obj = guided_tour_history)
#'obj2 <- list(type = "1d_tour", obj = tour_history_1d)
#'obj3 <- list(type = "scatter", obj = c("tars1", "tars2"))
#'obj4 <- list(type = "hist", obj = "head")
#'
#'interactive_tour(f,
#'                 col_names,
#'                list(obj1,obj2,obj3,obj4),
#'                 half_range,
#'                 n_max_cols=2)

interactive_tour <- function(data, col_names, plot_objects, half_range, n_max_cols, local=FALSE){
  pytourr_dir <- find.package("pytourr", lib.loc=NULL, quiet = TRUE)
  if (local==TRUE){
    pytourr_dir <- base::paste(pytourr_dir,"/inst/python", sep = "")
  } else {
    pytourr_dir <- base::paste(pytourr_dir,"/python", sep = "")
  }
  req_py_func <- "/interactive_tour.py"
  func_loc <- base::paste(pytourr_dir,req_py_func, sep = "")
  reticulate::source_python(func_loc)
  py$interactive_tour(data, col_names, plot_objects, half_range, n_max_cols)
}



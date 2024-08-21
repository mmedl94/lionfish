#' R wrapper for the interactive_tour function written on python
#'
#' @param data the dataset you want to investigate
#' @param plot_objects a named list of objects you want to be displayed. Each entry requires a definition of
#' the type of display and a specification of what should be plotted.
#' @param feature_names names of the features of the dataset
#' @param half_range factor that influences the scaling of the displayed tour plots.
#' Small values lead to more spread out datapoints (that might not fit the plotting area),
#' while large values lead to the data being more compact. If not provided a good estimate
#' will be calculated and used.
#' @param n_plot_cols specifies the number of columns of the grid of the final display.
#' @param preselection a vector that specifies in which subset each datapoint should be put initially.
#' @param preselection_names a vector that specifies the names of the preselection subsets
#' @param n_subsets the total number of available subsets (up to 10).
#' @param display_size rough size of each subplot in inches
#'
#' @return -
#' @export
#'
#' @examples
#'
#'library(tourr)
#'library(reticulate)
#'library(pytourr)
#'
#'data <- apply(flea[,1:6], 2, function(x) (x-mean(x))/sd(x))
#'clusters <- as.numeric(flea$species)
#'flea_subspecies <- unique(flea$species)
#'
#'guided_tour_history <- save_history(data,
#'                                    tour_path = guided_tour(holes()))
#'grand_tour_history_1d <- save_history(data,
#'                                      tour_path = grand_tour(d=1))
#'
#'half_range <- max(sqrt(rowSums(f^2)))
#'feature_names <- colnames(f)
#'
#'init_env()
#'
#'obj1 <- list(type = "2d_tour", obj = guided_tour_history)
#'obj2 <- list(type = "1d_tour", obj = grand_tour_history_1d)
#'obj3 <- list(type = "scatter", obj = c("tars1", "tars2"))
#'obj4 <- list(type = "hist", obj = "head")
#'
#'interactive_tour(data=data,
#'                 plot_objects=list(obj1, obj2, obj3, obj4),
#'                 feature_names=feature_names,
#'                 half_range = half_range,
#'                 n_plot_cols=2,
#'                 preselection=clusters,
#'                 preselection_names=flea_subspecies,
#'                 n_subsets = 5
#'                 display_size=5)

interactive_tour <- function(data, plot_objects, feature_names, half_range,
                             n_plot_cols, preselection=FALSE,
                             preselection_names=FALSE, n_subsets=3, display_size=5){

  pytourr_dir <- find.package("pytourr", lib.loc=NULL, quiet = TRUE)

  if (dir.exists(file.path(pytourr_dir, "/inst"))){
    pytourr_dir <- base::paste(pytourr_dir,"/inst/python", sep = "")
  } else {
    pytourr_dir <- base::paste(pytourr_dir,"/python", sep = "")
  }
  req_py_func <- "/interactive_tour.py"

  func_loc <- base::paste(pytourr_dir,req_py_func, sep = "")
  reticulate::source_python(func_loc)
  reticulate::py$interactive_tour(data, plot_objects, feature_names, half_range,
                                  n_plot_cols, preselection,
                                  preselection_names, n_subsets, display_size)
}



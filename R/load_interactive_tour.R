#'  R wrapper for the load_interactive_tour function written on python
#'  It allows a user to load a previously saved interactive tour snapshot
#'
#' @param data the dataset you want to investigate. Must be the same as the
#' dataset that was loaded when the save was created!
#' @param directory_to_save path to the location of the save folder
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
#' @param hover_cutoff number of features at which the switch from intransparent
#' to transparent labels that can be hovered over to make them intransparent occurs
#' @param label_size size of the labels of the feature names of 1d and 2d tours
#' @param axes_blendout_threshhold initial value of the threshold for blending
#' out projection axes with a smaller length
#'
#' @return
#' @export
#'
#' @examples
#'
load_interactive_tour <- function(data, directory_to_save,
                                  feature_names=NULL, half_range=NULL,
                                  n_plot_cols=2, preselection=FALSE,
                                  preselection_names=FALSE, n_subsets=FALSE,
                                  display_size=5,hover_cutoff=10,
                                  label_size=15, axes_blendout_threshhold=1){

  pytourr_dir <- find.package("lionfish", lib.loc=NULL, quiet = TRUE)

  if (dir.exists(file.path(pytourr_dir, "/inst"))){
    pytourr_dir <- base::paste(pytourr_dir,"/inst/python", sep = "")
  } else {
    pytourr_dir <- base::paste(pytourr_dir,"/python", sep = "")
  }
  req_py_func <- "/interactive_tour.py"

  if (is.null(feature_names)){
    feature_names <- paste("feature", 1:ncol(data))
  }

  if (file.exists(paste0(directory_to_save,"/attributes.pkl"))){
    print(paste0("loading from ", directory_to_save))
  } else if(file.exists(paste0(getwd(), directory_to_save,"/attributes.pkl"))){
    directory_to_save <- paste0(getwd(), directory_to_save)
    print(paste0("loading from ", directory_to_save))
  } else if(file.exists(paste0(getwd(),"/", directory_to_save,"/attributes.pkl"))){
    directory_to_save <- paste0(getwd(),"/", directory_to_save)
    print(paste0("loading from ", directory_to_save))
  } else {
    print(paste0("loading from ",getwd(), directory_to_save))
  }

  func_loc <- base::paste(pytourr_dir,req_py_func, sep = "")
  reticulate::source_python(func_loc)
  reticulate::py$load_interactive_tour(data, directory_to_save, feature_names,
                                       half_range, n_plot_cols, preselection,
                                       preselection_names, n_subsets,
                                       display_size,hover_cutoff, label_size,
                                       axes_blendout_threshhold)
  }

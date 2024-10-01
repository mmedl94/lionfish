#'  R wrapper for the load_interactive_tour function written on python
#'  It allows a user to load a previously saved interactive tour snapshot
#'
#' @param data
#' @param directory_to_save
#' @param feature_names
#' @param half_range
#' @param n_plot_cols
#' @param preselection
#' @param preselection_names
#' @param n_subsets
#' @param display_size
#' @param hover_cutoff
#' @param label_size
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

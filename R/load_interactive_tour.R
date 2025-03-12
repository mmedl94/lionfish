#' @title R Wrapper for 'load_interactive_tour' Function Written in 'python'
#' @description
#' Loads a previously saved snapshot created by pressing the "Save projections
#' and subsets" within the GUI. The data that was loaded when saving a
#' snapshot has to be provided to this function when loading that snapshot.
#' Additionally, this function allows to adjust some parameters of the GUI when
#' loading a snapshot, such as display_size or label_size.
#' @param data the dataset you want to investigate. Must be the same as the
#' dataset that was loaded when the save was created!
#' @param directory_to_save path to the location of the save folder
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
#' @param color_scale a viridis/matplotlib colormap to define the color scheme of the subgroups
#' @param color_scale_heatmap a viridis/matplotlib colormap to define the color scheme of the heatmap
#' @param axes_blendout_threshhold initial value of the threshold for blending
#' out projection axes with a smaller length
#'
#' @return opens the interactive GUI
#'
#' @export
#'
#' @examples
#' data("flea", package = "tourr")
#' data <- flea[1:6]
#' if (check_env()){
#' init_env()
#' pytourr_dir <- find.package("lionfish", lib.loc = NULL, quiet = TRUE)
#' pytourr_dir <- paste(pytourr_dir, "/inst/test_snapshot", sep = "")
#' if (interactive()){
#' load_interactive_tour(data, pytourr_dir)
#' }
#' }

load_interactive_tour <- function(data, directory_to_save, half_range = NULL,
                                  n_plot_cols = 2, preselection = FALSE,
                                  preselection_names = FALSE, n_subsets = FALSE,
                                  display_size = 5, hover_cutoff = 10,
                                  label_size = 15, color_scale = "default",
                                  color_scale_heatmap="default",
                                  axes_blendout_threshhold = 1) {
  pytourr_dir <- find.package("lionfish", lib.loc = NULL, quiet = TRUE)
  feature_dir <- base::paste(pytourr_dir, "/inst/test_snapshot/feature_selection.csv", sep = "")
  feature_names <- utils::read.csv(feature_dir, header = FALSE)$V1

  if (dir.exists(file.path(pytourr_dir, "/inst"))) {
    pytourr_dir <- base::paste(pytourr_dir, "/inst/python", sep = "")
  } else {
    pytourr_dir <- base::paste(pytourr_dir, "/python", sep = "")
  }
  req_py_func <- "/interactive_tour.py"

  if (file.exists(paste0(directory_to_save, "/attributes.pkl"))) {
    base::message(paste0("loading from ", directory_to_save))
  } else if (file.exists(paste0(getwd(), directory_to_save, "/attributes.pkl"))) {
    directory_to_save <- paste0(getwd(), directory_to_save)
    base::message(paste0("loading from ", directory_to_save))
  } else if (file.exists(paste0(getwd(), "/", directory_to_save, "/attributes.pkl"))) {
    directory_to_save <- paste0(getwd(), "/", directory_to_save)
    base::message(paste0("loading from ", directory_to_save))
  } else {
    base::message(paste0("loading from ", getwd(), directory_to_save))
  }

  func_loc <- base::paste(pytourr_dir, req_py_func, sep = "")
  reticulate::source_python(func_loc)
  reticulate::py$load_interactive_tour(
    data, directory_to_save, feature_names,
    half_range, n_plot_cols, preselection,
    preselection_names, n_subsets,
    display_size, hover_cutoff, label_size,
    color_scale, color_scale_heatmap,
    axes_blendout_threshhold
  )
}

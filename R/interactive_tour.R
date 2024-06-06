interactive_tour <- function(data, col_names, plot_objects, half_range, n_max_cols){
  pytourr_dir <- find.package("pytourr", lib.loc=NULL, quiet = TRUE)
  pytourr_dir <- base::paste(pytourr_dir,"/python", sep = "")
  req_py_func <- "/interactive_tour.py"
  func_loc <- base::paste(pytourr_dir,req_py_func, sep = "")
  reticulate::source_python(func_loc)
  py$interactive_tour(data, col_names, plot_objects, half_range, n_max_cols)
}



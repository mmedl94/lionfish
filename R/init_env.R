#' Initialize anaconda environment used for python backend
#'
#' @param env_name A string that defines the name of the anaconda environment reticulate uses
#'
#' @return -
#' @export
#'
#' @examples init_env(env_name="r-pytourr")
#'
init_env <- function(env_name="r-pytourr", virtual_env = "virtual_env", local=FALSE){

  # Check if python is available
  reticulate::py_available(initialize = FALSE)

  if (virtual_env == "anaconda"){
    # check if python environment exists and create new one if not
    if (env_name %in% reticulate::conda_list()$name==FALSE){
      reticulate::conda_create(env_name)
    }
    # initialize python environment
    reticulate::use_condaenv(env_name)

    # check if required packages are installed and install them if not
    package_names <- reticulate::py_list_packages(envname = env_name, "conda")
    print(package_names)
    required_packages <- c("pandas", "numpy", "matplotlib", "customtkinter")
    for (package in required_packages){
      if (package %in% package_names$package==FALSE){
        if (package == "customtkinter"){
          reticulate::conda_install(env_name, package, pip=TRUE)
        } else {
          reticulate::conda_install(env_name, package)
        }
      }
    }
  } else if (virtual_env == "virtual_env"){
    if (env_name %in% reticulate::virtualenv_list()==FALSE){
      py_version <- unlist(reticulate::virtualenv_starter(all=TRUE)$version[1])
      py_version = paste(py_version, collapse = ".")
      reticulate::install_python(version = py_version)
      reticulate::virtualenv_create(env_name)
    }
    # initialize python environment
    reticulate::use_virtualenv(env_name)

    # check if required packages are installed and install them if not
    package_names <- reticulate::py_list_packages(envname = env_name, "virtualenv")
    required_packages <- c("pandas", "numpy", "matplotlib", "customtkinter")
    for (package in required_packages){
      if (package %in% package_names$package==FALSE){
        if (package == "customtkinter"){
          reticulate::virtualenv_install(env_name, package)
        } else {
          reticulate::virtualenv_install(env_name, package)
        }
      }
    }
  }

  base::cat(base::sprintf('Python environment "%s" successfully loaded', env_name), "\n")

  # Check accessibility of python functions
  pytourr_dir <- find.package("pytourr", lib.loc=NULL, quiet = TRUE)

  if (dir.exists(file.path(pytourr_dir, "/inst"))){
    check_dir <- base::paste(pytourr_dir,"/inst/python/check_pytour.py", sep = "")
  } else {
    check_dir <- base::paste(pytourr_dir,"/python/check_pytour.py", sep = "")
  }
  reticulate::source_python(check_dir)
  reticulate::py$check_pytour()
}

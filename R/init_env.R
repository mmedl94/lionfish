#' Initialize anaconda environment used for python backend
#'
#' @param env_name a string that defines the name of the python environment reticulate uses.
#' This can be useful if one wants to use a preinstalled python environment.
#' @param virtual_env either "virtual_env" or "anaconda". "virtual_env" creates
#' a virtual environment, which has the advantage that the GUI looks much nicer
#' and no previous python installation is required,but the setup of the
#' environment can be more error prone. "anaconda" installs the python
#' environment via Anaconda, which can be more stable, but the GUI looks more
#' dated.
#'
#' @return -
#' @export
#'
#' @examples init_env(env_name="r-lionfish", virtual_env = "virtual_env")
#'
init_env <- function(env_name="r-lionfish", virtual_env = "virtual_env", local=FALSE){

  # Check if python is available
  reticulate::py_available(initialize = FALSE)

  required_packages <- c("pandas", "numpy", "matplotlib",
                         "customtkinter", "statsmodels", "seaborn")

  if (virtual_env == "anaconda"){
    # check if python environment exists and create new one if not
    if (env_name %in% reticulate::conda_list()$name==FALSE){
      reticulate::conda_create(env_name)
    }
    # initialize python environment
    reticulate::use_condaenv(env_name)

    # check if required packages are installed and install them if not
    package_names <- reticulate::py_list_packages(envname = env_name, "conda")
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
  lionfish_dir <- find.package("lionfish", lib.loc=NULL, quiet = TRUE)

  if (dir.exists(file.path(lionfish_dir, "/inst"))){
    check_dir <- base::paste(lionfish_dir,"/inst/python/check_backend.py", sep = "")
  } else {
    check_dir <- base::paste(lionfish_dir,"/python/check_backend.py", sep = "")
  }
  reticulate::source_python(check_dir)
  reticulate::py$check_backend()

  # set directory of tcl/tk installation for windows users
  if (.Platform$OS.type == "windows"){
    home_dir <- gsub("\\\\", "/", Sys.getenv("USERPROFILE"))

    sys <- import("sys")
    full_python_version <- sys$version
    py_version <- strsplit(full_python_version, " ")[[1]][1]

    py_dir <- "/AppData/Local/r-reticulate/r-reticulate/pyenv/pyenv-win/versions/"
    tk <- "/tk/tk8.6"
    tcl <- "/tcl/tcl8.6"

    tcl_dir <- base::paste0(home_dir,py_dir, py_version, tcl)
    tk_dir <-  base::paste0(home_dir,py_dir, py_version, tk)

    Sys.setenv(TCL_LIBRARY = tcl_dir)
    Sys.setenv(TK_LIBRARY = tk_dir)

    cat("tcl directory", tcl_dir, "\n")
    cat("tk directory", tk_dir, "\n")
  }
}

#' @title Check Whether 'anaconda' Environment Exists
#' @description
#' Checks if 'anaconda' environment of the given name is installed and returns
#' TRUE if so.
#' @param env_name a string that defines the name of the 'anaconda' environment
#' reticulate uses.
#' @return boolean
#' @export
#'
#' @examples
#' check_conda_env(env_name="r-lionfish")
check_conda_env <- function(env_name="r-lionfish"){
  conda_bin <- tryCatch(
    reticulate::conda_binary(),
    error = function(e) NULL
  )
  if (!is.null(conda_bin)){
    if (reticulate::condaenv_exists(env_name)){
      return(TRUE)
    }
  }
  return(FALSE)
}

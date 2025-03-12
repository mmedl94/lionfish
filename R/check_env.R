#' @title Check Whether 'python' Environment Exists
#' @description
#' Checks whether 'python' environment of a given name exists an returns TRUE
#' if it does. Also checks if 'anaconda' is installed and catches the error
#' if it isn't, but returns FALSE.
#' @param env_name a string that defines the name of the 'python' environment
#' reticulate uses.
#' @return boolean
#' @export
#'
#' @examples
#' check_env(env_name="r-lionfish")
check_env <- function(env_name="r-lionfish"){

  if (env_name %in% reticulate::virtualenv_list()){
    return(TRUE)
  }

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

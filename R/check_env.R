#' @title Check Whether 'python' Environment Exists
#' @description
#' Checks whether 'python' environment of a given name exists an returns TRUE
#' if it does.
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

  if (!is.null(reticulate::conda_binary())){
    if (reticulate::condaenv_exists(env_name)){
      return(TRUE)
    }
  }

  return(FALSE)
}

#' @title Check Whether Virtual 'python' Environment Exists
#' @description
#' Checks whether virtual 'python' environment of a given name exists an returns
#' TRUE if it does.
#' @param env_name a string that defines the name of the 'python' environment
#' reticulate uses.
#' @return boolean
#' @export
#'
#' @examples
#' check_venv(env_name="r-lionfish")
check_venv <- function(env_name="r-lionfish"){
  if (env_name %in% reticulate::virtualenv_list()){return(TRUE)}
  return(FALSE)
}

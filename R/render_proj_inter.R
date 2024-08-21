#' Modification of the render_proj() function of tourr so that the half_range is calculated by
#' max(sqrt(rowSums(data^2))) or can be provided as arg
#'
#' @param data matrix, or data frame containing numeric columns,
#'   should be standardized to have mean 0, sd 1
#' @param prj projection matrix
#' @param axis_labels of the axes to be displayed
#' @param obs_labels labels of the observations to be available for interactive mouseover
#' @param limits value setting the lower and upper limits of
#'   projected data, default 1
#' @param position position of the axes: center (default),
#'   bottomleft or off
#'
#' @return list containing projected data, circle and segments for axes
#' @export
#'
#' @examples
#' library(tourr)
#' data(flea)
#' flea_std <- apply(flea[,1:6], 2, function(x) (x-mean(x))/sd(x))
#' prj <- basis_random(ncol(flea[,1:6]), 2)
#' p <- render_proj(flea_std, prj)
#'
render_proj_inter <- function(data, prj, half_range=NULL, axis_labels=NULL, obs_labels=NULL, limits=1, position="center"){
  # Check dimensions ok
  try(if (ncol(data) != nrow(prj))
    stop("Number of columns of data don't match number of rows of prj"))
  try(if(ncol(prj) != 2)
    stop("Number of columns of prj needs to be 2"))

  # Project data and scale into unit box
  data_prj <- as.matrix(data) %*% as.matrix(prj)
  if (is.null(half_range)){
    half_range <- max(sqrt(rowSums(data_prj^2)))
  }
  data_prj <- data_prj/half_range
  colnames(data_prj) <- c("P1", "P2")
  data_prj <- data.frame(data_prj)

  # Add observation labels
  if (is.null(obs_labels))
    obs_labels <- as.character(1:nrow(data))
  data_prj$obs_labels <- obs_labels

  # Axis scale
  if (position == "center") {
    axis_scale <- 2 * limits / 3
    axis_pos <- 0
  } else if (position == "bottomleft") {
    axis_scale <- limits / 6
    axis_pos <- -2 / 3 * limits
  }
  adj <- function(x) axis_pos + x * axis_scale

  # Compute segments
  axes <- data.frame(x1=adj(0), y1=adj(0),
                     x2=adj(prj[, 1]), y2=adj(prj[, 2]))
  # Make labels if missing
  if (is.null(axis_labels))
    axis_labels <- colnames(data)
  rownames(axes) <- axis_labels

  # Compute circle
  theta <- seq(0, 2 * pi, length = 50)
  circle <- data.frame(c1 = adj(cos(theta)), c2=adj(sin(theta)))

  return(list(data_prj=data_prj, axes=axes, circle=circle))
}

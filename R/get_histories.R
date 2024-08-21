#' get local history
#'
#' Function for python backend to receive a local tour based on current projection
#'
#' @param data the dataset to calculate the projections with. In practice
#' only the two first rows of the dataset are provided as the actual data is not
#' needed.
#' @param starting_projection the initial projection one wants to initiate the
#' local tour from
#'
#' @return history object containing the projections of the requested tour
#' @export
#'
#' @examples .
#'
get_local_history <- function(data, starting_projection){
  history <- save_history(data,
                          tour_path = local_tour(starting_projection))
  return(history)
}

#' get guided tour-holes history
#'
#' Function for python backend to receive a guided tour-holes
#'
#' @param data the dataset to calculate the projections with.
#' @param dimension 1 for a 1d tour or 2 for a 2d tour
#'
#' @return history object containing the projections of the requested tour
#' @export
#'
#' @examples .
#'
get_guided_holes_history <- function(data, dimension){
  history <- save_history(data,
                          tour_path = guided_tour(holes(),
                                                  d=dimension))
  return(history)
}

#' get guided tour-holes-better history
#'
#' Function for python backend to receive a guided tour-holes that searches for
#' a better projection near the current projection.
#'
#' @param data the dataset to calculate the projections with.
#' @param dimension 1 for a 1d tour or 2 for a 2d tour
#'
#' @return history object containing the projections of the requested tour
#' @export
#'
#' @examples .
#'
get_guided_holes_better_history <- function(data, dimension){
  history <- save_history(data,
                          tour_path = guided_tour(holes(),
                                                  d = dimension,
                                                  search_f = search_better))
  return(history)
}

#' get guided tour-lda history
#'
#' Function for python backend to receive a guided tour-lda history
#'
#' @param data the dataset to calculate the projections with
#' @param clusters the clusters for the lda to be performed on
#' @param dimension 1 for a 1d tour or 2 for a 2d tour
#'
#' @return history object containing the projections of the requested tour
#' @export
#'
#' @examples .
#'
get_guided_lda_history <- function(data, clusters, dimension){
  history <- save_history(data,
                          tour_path = guided_tour(lda_pp(clusters),
                                                  d=dimension))
  return(history)
}

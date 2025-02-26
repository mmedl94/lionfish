#' Get Local Tour History
#'
#' Function for python backend to receive a local tour based on current projection
#'
#' @title Get Local Tour History
#' @param data the dataset to calculate the projections with. In practice
#' only the two first rows of the dataset are provided as the actual data is not
#' needed.
#' @param starting_projection the initial projection one wants to initiate the
#' local tour from
#'
#' @return history object containing the projections of the requested tour
#' @export
#' @examples
#' library(tourr)
#' data(flea)
#' flea <- flea[-7]
#' prj <- basis_random(ncol(flea), 2)
#' get_local_history(flea, prj)
#'
get_local_history <- function(data, starting_projection){
  history <- tourr::save_history(data,
                          tour_path = tourr::local_tour(starting_projection))
  return(history)
}

#' Get Guided Tour-Holes History
#'
#' Function for python backend to receive a guided tour-holes
#' @title Get Guided Tour-Holes History
#'
#' @param data the dataset to calculate the projections with.
#' @param dimension 1 for a 1d tour or 2 for a 2d tour
#'
#' @return history object containing the projections of the requested tour
#' @export
#' @examples
#' library(tourr)
#' data(flea)
#' flea <- flea[-7]
#' get_guided_holes_history(flea, 2)
#'
get_guided_holes_history <- function(data, dimension){
  history <- tourr::save_history(data,
                          tour_path = tourr::guided_tour(tourr::holes(),
                                                  d=dimension))
  return(history)
}

#' Get Guided Tour-Holes-Better History
#'
#' Function for python backend to receive a guided tour-holes that searches for
#' a better projection near the current projection.
#'
#' @title Get Guided Tour-Holes-Better History
#' @param data the dataset to calculate the projections with.
#' @param dimension 1 for a 1d tour or 2 for a 2d tour
#'
#' @return history object containing the projections of the requested tour
#' @export
#'
#' @examples
#' library(tourr)
#' data(flea)
#' flea <- flea[-7]
#' get_guided_holes_better_history(flea, 2)
#'
get_guided_holes_better_history <- function(data, dimension){
  history <- tourr::save_history(data,
                          tour_path = tourr::guided_tour(tourr::holes(),
                                                  d = dimension,
                                                  search_f = tourr::search_better))
  return(history)
}

#' Get Guided Tour-LDA History
#'
#' Function for python backend to receive a guided tour-lda history
#' @title Get Guided Tour-LDA History
#' @param data the dataset to calculate the projections with
#' @param clusters the clusters for the lda to be performed on
#' @param dimension 1 for a 1d tour or 2 for a 2d tour
#'
#' @return history object containing the projections of the requested tour
#' @export
#' @examples
#' library(tourr)
#' data(flea)
#' clusters <- as.numeric(factor(flea[[7]]))
#' get_guided_lda_history(flea[-7], clusters, 2)
#'
get_guided_lda_history <- function(data, clusters, dimension){
  history <- tourr::save_history(data,
                          tour_path = tourr::guided_tour(tourr::lda_pp(clusters),
                                                  d=dimension))
  return(history)
}

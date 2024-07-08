get_local_history <- function(data, starting_projection){
  history <- save_history(data,
                          tour_path = local_tour(starting_projection))
  return(history)
}

get_guided_holes_history <- function(data, dimension){
  history <- save_history(data,
                          tour_path = guided_tour(holes(),
                          d=dimension))
  return(history)
}

get_guided_holes_better_history <- function(data, dimension){
  history <- save_history(data,
                          tour_path = guided_tour(holes(),
                            d = dimension,
                            search_f = search_better))
  return(history)
}

get_guided_lda_history <- function(data, clusters, dimension){
  history <- save_history(data,
                          tour_path = guided_tour(lda_pp(clusters),
                          d=dimension))
  return(history)
}

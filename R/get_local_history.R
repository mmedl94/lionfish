get_local_history <- function(data, starting_projection){
  local_history <- save_history(data,
                                tour_path = local_tour(starting_projection))
  return(local_history)
}

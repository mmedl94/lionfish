% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/interactive_tour.R
\name{interactive_tour}
\alias{interactive_tour}
\title{R wrapper for the interactive_tour function written on python}
\usage{
interactive_tour(
  data,
  plot_objects,
  feature_names = NULL,
  half_range = NULL,
  n_plot_cols = 2,
  preselection = FALSE,
  preselection_names = FALSE,
  n_subsets = 3,
  display_size = 5,
  hover_cutoff = 10,
  label_size = 15
)
}
\arguments{
\item{data}{the dataset you want to investigate}

\item{plot_objects}{a named list of objects you want to be displayed. Each entry requires a definition of
the type of display and a specification of what should be plotted.}

\item{feature_names}{names of the features of the dataset}

\item{half_range}{factor that influences the scaling of the displayed tour plots.
Small values lead to more spread out datapoints (that might not fit the plotting area),
while large values lead to the data being more compact. If not provided a good estimate
will be calculated and used.}

\item{n_plot_cols}{specifies the number of columns of the grid of the final display.}

\item{preselection}{a vector that specifies in which subset each datapoint should be put initially.}

\item{preselection_names}{a vector that specifies the names of the preselection subsets}

\item{n_subsets}{the total number of available subsets (up to 10).}

\item{display_size}{rough size of each subplot in inches}

\item{hover_cutoff}{number of features at which the switch from intransparent
to transparent labels that can be hovered over to make them intransparent occurs}

\item{label_size}{size of the labels of the feature names of 1d and 2d tours}
}
\value{
\itemize{
\item 
}
}
\description{
R wrapper for the interactive_tour function written on python
}
\examples{
library(tourr)
library(reticulate)
library(lionfish)

data <- apply(flea[,1:6], 2, function(x) (x-mean(x))/sd(x))
clusters <- as.numeric(flea$species)
flea_subspecies <- unique(flea$species)

guided_tour_history <- save_history(data,
                                   tour_path = guided_tour(holes()))
grand_tour_history_1d <- save_history(data,
                                     tour_path = grand_tour(d=1))

half_range <- max(sqrt(rowSums(f^2)))
feature_names <- colnames(f)

init_env()

obj1 <- list(type = "2d_tour", obj = guided_tour_history)
obj2 <- list(type = "1d_tour", obj = grand_tour_history_1d)
obj3 <- list(type = "scatter", obj = c("tars1", "tars2"))
obj4 <- list(type = "hist", obj = "head")

interactive_tour(data=data,
                plot_objects=list(obj1, obj2, obj3, obj4),
                feature_names=feature_names,
                half_range = half_range,
                n_plot_cols=2,
                preselection=clusters,
                preselection_names=flea_subspecies,
                n_subsets = 5,
                display_size=5)
}

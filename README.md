# lionfish: an expLoratory Interactive tOol for dyNamic visualization and identiFicatIon multidimenSional mecHanisms

This package is an extension of the [tourr](https://github.com/ggobi/tourr) R-package. For a general overview of the tourr package please refer to the [tourr documentation](https://ggobi.github.io/tourr/). lionfish adds interactive displays to the functionality of tourr allowing users to direct the path of the tours.

## Installation

You can install the development version of lionfish from github with:

``` r
install.packages("remotes")
remotes::install_github("mmedl94/lionfish")
```

Make sure you have git installed. You can download and install git from <https://git-scm.com/downloads>.

### Complications for windows users

Running the example code below might result in the following error

``` r
Error: Required version of NumPy not available: incompatible NumPy binary version 33554432 (expecting version 16777225)
```

To resolve this, we have to delete the erroneous virtual environment and build a new one with an older Python version.

``` r
# Delete the old enironment, might require a restart of R
reticulate::virtualenv_remove("r-lionfish")

# Install the stable Python version
reticulate::install_python(version="3.8.10")

# Build new virtual environment
reticulate::virtualenv_create(envname = "r-lionfish", version="3.8.10")

# Initiate Python environment as usual
init_env()
```

## Example

To run an interactive tour you will first have to initialize the python backend with

``` r
library(lionfish)
init_env()
```

Then you can display saved tour objects, scatterplots or histograms with interactive_tour()

``` r
library(tourr)
data <- apply(flea[,1:6], 2, function(x) (x-mean(x))/sd(x))
clusters <- as.numeric(flea$species)
flea_subspecies <- unique(flea$species)

guided_tour_history <- save_history(data,
                                    tour_path = guided_tour(holes()))
grand_tour_history_1d <- save_history(data,
                                      tour_path = grand_tour(d=1))

half_range <- max(sqrt(rowSums(data^2)))
feature_names <- colnames(data)

init_env()

obj1 <- list(type="2d_tour", obj=guided_tour_history)
obj2 <- list(type="1d_tour", obj=grand_tour_history_1d)
obj3 <- list(type="scatter", obj=c("tars1", "tars2"))
obj4 <- list(type="hist", obj="head")

interactive_tour(data=data,
                 plot_objects=list(obj1, obj2, obj3, obj4),
                 feature_names=feature_names,
                 half_range=half_range,
                 n_plot_cols=2,
                 preselection=clusters,
                 preselection_names=flea_subspecies,
                 n_subsets=5,
                 display_size=5)
```

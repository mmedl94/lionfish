pytourr: extension of the tourr package tour for interactive display
================

This package is an extension of the [tourr](https://github.com/ggobi/tourr) R-package.
For a general overview of the tourr package please refer to the 
[tourr documentation](https://ggobi.github.io/tourr/). pytourr adds interactive displays
to the functionality of tourr allowing users to direct the path of the tours.

## Installation

You can install the development version of pytourr from github with:

``` r
install.packages("remotes")
remotes::install_github("mmedl94/pytourr")
```

Make sure you have git installed. You can download and install git 
from [https://git-scm.com/downloads](https://git-scm.com/downloads) if you
don't have it installed.


###  Complications for windows users

Running the example code below might result in the following error
``` r
Error: Required version of NumPy not available: incompatible NumPy binary version 33554432 (expecting version 16777225)
```

To resolve this, we have to delete the erroneous virtual environment, build a new one with an older Python version and
provide the directories to our tcl and tk installations.

``` r
# Delete the old enironment, might require a restart of R
reticulate::virtualenv_remove("r-pytourr")

# Install the stable Python version
reticulate::install_python(version="3.8.10")

# Build new virtual environment
reticulate::virtualenv_create(envname = "r-pytourr", version="3.8.10")

# Initiate Python environment as usual
init_env()
```

Running the example will now probably throw:
``` r
Error in py_call_impl(callable, call_args$unnamed, call_args$named) : 
  _tkinter.TclError: Can't find a usable init.tcl in the following directories:
    C:/some/directory/

This probably means that Tcl wasn't installed properly.
```
This means that tkinter cannot find your tk and tcl installations. So we have to provide the correct directories manually.

They should be located somewhere here (please modify according to your system!)
``` r
# tcl location
C:/Users/user/AppData/Local/r-reticulate/r-reticulate/pyenv/pyenv-win/versions/3.8.10/tcl/tcl8.6

# tk location
C:/Users/user/AppData/Local/r-reticulate/r-reticulate/pyenv/pyenv-win/versions/3.8.10/tk/tk8.6
```
Once you have located them please run (please modify according to your system!)
``` r
Sys.setenv(TCL_LIBRARY = "C:/Users/user/AppData/Local/r-reticulate/r-reticulate/pyenv/pyenv-win/versions/3.8.10/tcl/tcl8.6")
Sys.setenv(TK_LIBRARY = "C:/Users/user/AppData/Local/r-reticulate/r-reticulate/pyenv/pyenv-win/versions/3.8.10/tk/tk8.6")
```
The directories to the tcl and tk installations have to be set anew 
at the beginning of each R session.

## Example

To run an interactive tour you will first have to initialize the python backend with 

``` r
library(pytourr)
init_env()
```
Then you can display saved tour objects, scatterplots or histograms with interactive_tour()

``` r
library(tourr)
library(reticulate)
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
```

pytourr: extension of the tourr package tour for interactive displays
================

# pytourr

This package is an extension of the [tourr](https://github.com/ggobi/tourr) R-package.
For a general overview of the tourr package please refer to the 
[tourr documentation](https://ggobi.github.io/tourr/). pytourr adds interactive displays
to the functionality of tourr allowing users to direct the path of the tours.

## Installation

You can install the development version of pytourr from github with:

``` r
# install.packages("remotes")
remotes::install_github("mmedl94/pytourr/")
```

## Example

To run an interactive tour you will first have to initialize the python backend with 

``` r
init_env()
```
Then you can display saved tour objects, scatterplots or histograms with interactive_tour()

``` r
library(tourr)
data <- apply(flea[,1:6], 2, function(x) (x-mean(x))/sd(x))

guided_tour_history <- save_history(data, 
                                    tour_path = guided_tour(holes()))
grand_tour_history_1d <- save_history(data, 
                                tour_path = grand_tour(d=1))

half_range <- max(sqrt(rowSums(data^2)))
col_names <- colnames(data)

obj1 <- list(type = "2d_tour", obj = guided_tour_history)
obj2 <- list(type = "1d_tour", obj = grand_tour_history_1d)
obj3 <- list(type = "scatter", obj = c("tars1", "tars2"))
obj4 <- list(type = "hist", obj = "head")

interactive_tour(data,
                 col_names,
                 list(obj1,obj2,obj3,obj4),
                 half_range,
                 n_max_cols=2)

```

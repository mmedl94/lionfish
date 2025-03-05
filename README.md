# lionfish: an expLoratory Interactive tOol for dyNamic visualization and identiFicatIon multidimenSional mecHanisms

This package is an extension of the [tourr](https://github.com/ggobi/tourr) R-package. For a general overview of the tourr package please refer to the [tourr documentation](https://ggobi.github.io/tourr/). lionfish adds interactive displays to the functionality of tourr allowing users to direct the path of the tours.

## lionfish documentation

For an overview of the functionality of lionfish please visit the [documentation](https://mmedl94.github.io/lionfish/)!

## Installation

You can install the development version of lionfish from github with:

``` r
install.packages("remotes")
remotes::install_github("mmedl94/lionfish")
```

Make sure you have git installed. You can download and install git from <https://git-scm.com/downloads>.

Complications may arise when installing and accessing the Python backend of this package. If you run into any, please don't refrain from opening an issue!

## Known issues

For an unknown reason, the GUI sometimes appears empty at the initial launch.
Restarting R and launching the GUI again seems to resolve this.

## Example

In this introductory example we will launch the lionfish GUI with the 'flea'
datset and a selection of interactive display types. For more information about
the 'flea' dataset visit the
[tourr documentation](https://ggobi.github.io/tourr/reference/Flea-measurements.html).
In brief, the dataset contains six numeric variables with measurements of the fleas'
appendages and a factor variable with the flea species (Concinna, Heptapot. or 
Heikert.).
For more information on how to use the GUI visit the
[lionfish documentation](https://mmedl94.github.io/lionfish/).

First, we have to load the package and initialize the 'python' backend with:

``` r
library(lionfish)
init_env()
```

We first fetch the flea dataset from 'tourr', save the species column separately,
and get the unique species names as well as the feature names of our dataset.
Then we run and store a 'guided_tour' with the 'holes' index and a 'grand_tour'.
More information on tours can be found [here](https://ggobi.github.io/tourr/).

``` r
library(tourr)
data("flea")
data <- flea[1:6]
clusters <- as.numeric(flea$species)
flea_subspecies <- unique(flea$species)
feature_names <- colnames(data)

guided_tour_history <- save_history(data,
                                    tour_path = guided_tour(holes()))
grand_tour_history_1d <- save_history(data,
                                      tour_path = grand_tour(d=1))
```

Now we have to specify which display types we want to load into the GUI. In this
example, we want to display a two-dimensional tour, a one-dimensional tour, a
scatterplot, and a histogram. For each display type, we have to further specify
some settings as part of the so called plot objects. For the tours, these are
the stored tour histories, and for the scatterplot and histogram, we have to
define which features we want to plot. A detailed description of the plot
objects can be found [here](https://mmedl94.github.io/lionfish/articles/Plot-objects.html).

``` r
obj1 <- list(type="2d_tour", obj=guided_tour_history)
obj2 <- list(type="1d_tour", obj=grand_tour_history_1d)
obj3 <- list(type="scatter", obj=c("tars1", "tars2"))
obj4 <- list(type="hist", obj="head")
```

Finally, we launch the GUI with the 'interactive_tour' function. Details on the
interactive functions of the displays can be found
[here](https://mmedl94.github.io/lionfish/articles/General-interactivity.html).

``` r
interactive_tour(data=data,
                 plot_objects=list(obj1, obj2, obj3, obj4),
                 feature_names=feature_names,
                 n_plot_cols=2,
                 preselection=clusters,
                 preselection_names=flea_subspecies,
                 n_subsets=5,
                 display_size=5)
```

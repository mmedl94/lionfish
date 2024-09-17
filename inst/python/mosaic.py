
import numpy as np
import pandas as pd
from itertools import product
from statsmodels.graphics.mosaicplot import mosaic


def launch_mosaic(parent, plot_object, subplot_idx):
    mosaic_data = np.empty(
        (len(parent.feature_selection), int(parent.n_subsets)))
    non_empty_sets = []
    for subset_idx, subset in enumerate(parent.subselections):
        if subset.shape[0] != 0:
            mosaic_data[:,
                        subset_idx] = parent.data[subset].sum(axis=0)
            non_empty_sets.append(True)
        else:
            mosaic_data[:, subset_idx] = np.zeros(
                len(parent.feature_selection))
            non_empty_sets.append(False)
    mosaic_data = mosaic_data[parent.feature_selection]
    mosaic_data = mosaic_data[:, non_empty_sets]
    y_tick_labels = np.array(parent.feature_names)[parent.feature_selection]
    x_tick_labels = np.array([subselection_var.get()
                              for subselection_var in parent.subset_names])
    x_tick_labels = x_tick_labels[non_empty_sets]
    if plot_object["obj"] == "subgroups_on_y":
        tuples = list(
            product(y_tick_labels, x_tick_labels))
    else:
        tuples = list(
            product(x_tick_labels, y_tick_labels))
    index = pd.MultiIndex.from_tuples(
        tuples, names=["first", "second"])
    mosaic_data = pd.Series(
        mosaic_data.flatten(), index=index)
    if parent.initial_loop is False:
        parent.axs[subplot_idx].clear()
        parent.axs[subplot_idx].set_in_layout(True)
    mosaic_colors = np.array(parent.colors)[non_empty_sets]
    color_dict = {}
    if plot_object["obj"] == "subgroups_on_y":
        unique_levels = mosaic_data.index.get_level_values(
            "second").unique()
        color_mapping = dict(zip(unique_levels, mosaic_colors))
        for idx in mosaic_data.index:
            # Get the color for the current second level value
            color = color_mapping[idx[1]]
            # Map the index tuple to its corresponding color
            color_dict[idx] = {"color": color}
    else:
        unique_levels = mosaic_data.index.get_level_values(
            "first").unique()
        color_mapping = dict(zip(unique_levels, mosaic_colors))
        for idx in mosaic_data.index:
            # Get the color for the current 'second' level value
            color = color_mapping[idx[0]]
            # Map the index tuple to its corresponding color
            color_dict[idx] = {"color": color}
    mosaic(mosaic_data,
           ax=parent.axs[subplot_idx],
           properties=color_dict,
           gap=0.01)
    xlabels = parent.axs[subplot_idx].get_xticklabels()
    parent.axs[subplot_idx].set_xticklabels(xlabels,
                                            rotation=25,
                                            ha="right")

    ylabels = parent.axs[subplot_idx].get_yticklabels()
    parent.axs[subplot_idx].set_yticklabels(ylabels,
                                            rotation=25,
                                            ha="right")
    # remove extra plots
    twinaxs = parent.axs[subplot_idx].twinx()
    remove_pos = twinaxs.get_position().bounds
    twinaxs.remove()
    for axs_idx, axs in enumerate(parent.fig.get_axes()):
        if axs.get_position().bounds == remove_pos:
            if axs != parent.axs[subplot_idx]:
                axs.remove()

    for text in parent.axs[subplot_idx].texts:
        text.remove()
    plot_dict = {"type": "mosaic",
                 "subtype": "mosaic",
                 "subplot_idx": subplot_idx,
                 "mosaic_data": mosaic_data}
    parent.plot_dicts[subplot_idx] = plot_dict

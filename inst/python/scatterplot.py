import numpy as np
from pytour_selectors import LassoSelect


def launch_scatterplot(parent, plot_object, subplot_idx):
    # get data
    col_index_x = parent.col_names.index(plot_object["obj"][0])
    col_index_y = parent.col_names.index(plot_object["obj"][1])
    x = parent.data[:, col_index_x]
    y = parent.data[:, col_index_y]
    if parent.initial_loop is True:
        parent.fc = np.repeat(
            np.array(parent.colors[0])[:, np.newaxis], parent.n_pts, axis=1).T
        for idx, subset in enumerate(parent.subselections):
            if subset.shape[0] != 0:
                parent.fc[subset] = parent.colors[idx]
        scat = parent.axs[subplot_idx].scatter(x, y,
                                               animated=True)
        scat.set_facecolor(parent.fc)
    else:
        parent.axs[subplot_idx].collections[0].set_facecolors(
            parent.fc)
        parent.plot_dicts[subplot_idx]["selector"].disconnect()
    x_lims = parent.axs[subplot_idx].get_xlim()
    y_lims = parent.axs[subplot_idx].get_ylim()
    parent.axs[subplot_idx].set_xlim(x_lims)
    parent.axs[subplot_idx].set_ylim(y_lims)
    plot_dict = {"type": "scatter",
                 "subtype": "scatter",
                 "subplot_idx": subplot_idx
                 }
    parent.plot_dicts[subplot_idx] = plot_dict
    # start Lasso selector
    selector = LassoSelect(
        parent=parent,
        subplot_idx=subplot_idx
    )
    parent.plot_dicts[subplot_idx]["selector"] = selector
    x_name = plot_object["obj"][0]
    y_name = plot_object["obj"][1]
    parent.axs[subplot_idx].set_xlabel(x_name)
    parent.axs[subplot_idx].set_ylabel(y_name)
    parent.axs[subplot_idx].set_title(
        f"Scatterplot of variables {x_name} and {y_name}")

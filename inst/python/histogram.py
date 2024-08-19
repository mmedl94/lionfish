import numpy as np
from pytour_selectors import BarSelect


def launch_histogram(parent, plot_object, subplot_idx):
    if plot_object["obj"] in parent.col_names:
        col_index = parent.col_names.index(plot_object["obj"])
        x = parent.data[:, col_index]
        # clear old histogram
        parent.axs[subplot_idx].clear()
        # recolor preselected points
        x_subselections = []
        for subselection in parent.subselections:
            if subselection.shape[0] != 0:
                x_subselections.append(x[subselection])
            else:
                x_subselections.append(np.array([]))
        parent.axs[subplot_idx].hist(
            x_subselections,
            stacked=True,
            picker=True,
            color=parent.colors[:len(x_subselections)],
            animated=True)
        y_lims = parent.axs[subplot_idx].get_ylim()
        parent.axs[subplot_idx].set_ylim(y_lims)
        hist_variable_name = plot_object["obj"]
        parent.axs[subplot_idx].set_xlabel(hist_variable_name)
        parent.axs[subplot_idx].set_title(
            f"Histogram of variable {hist_variable_name}")
        parent.axs[subplot_idx].set_xticks([])
        if parent.initial_loop is True:
            parent.fc = np.repeat(
                np.array(parent.colors[0])[:, np.newaxis], parent.n_pts, axis=1).T
            for idx, subset in enumerate(parent.subselections):
                if subset.shape[0] != 0:
                    parent.fc[subset] = parent.colors[idx]
            plot_dict = {"type": "hist",
                         "subtype": "hist",
                         "subplot_idx": subplot_idx,
                         "hist_feature": col_index}
            parent.plot_dicts[subplot_idx] = plot_dict
            bar_selector = BarSelect(parent=parent,
                                     subplot_idx=subplot_idx)
            parent.plot_dicts[subplot_idx]["selector"] = bar_selector
        else:
            plot_dict = {"type": "hist",
                         "subtype": "hist",
                         "subplot_idx": subplot_idx,
                         "hist_feature": col_index,
                         "selector": parent.plot_dicts[subplot_idx]["selector"]}
            parent.plot_dicts[subplot_idx] = plot_dict
            parent.plot_dicts[subplot_idx]["selector"].disconnect(
            )
            bar_selector = BarSelect(parent=parent,
                                     subplot_idx=subplot_idx)
            parent.plot_dicts[subplot_idx]["selector"] = bar_selector
    else:
        print("Column not found")

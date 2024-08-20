
import numpy as np
import seaborn as sns


def launch_heatmap(parent, plot_object, subplot_idx):
    heatmap_data = np.empty(
        (len(parent.feature_selection), int(parent.n_subsets)))
    cur_metric_var = parent.metric_vars[subplot_idx].get()
    # get ratios
    all_pos = np.sum(parent.data, axis=0)
    non_empty_sets = []
    for subset_idx, subset in enumerate(parent.subselections):
        if subset.shape[0] != 0:
            non_empty_sets.append(True)
            all_pos_subset = np.sum(parent.data[subset], axis=0)
            if cur_metric_var == "Intra feature fraction":
                heatmap_data[:,
                             subset_idx] = all_pos_subset/parent.data[subset].shape[0]
            elif cur_metric_var == "Intra cluster fraction":
                heatmap_data[:,
                             subset_idx] = all_pos_subset/all_pos
            elif cur_metric_var == "Total fraction":
                heatmap_data[:,
                             subset_idx] = all_pos_subset/parent.data.shape[0]
        else:
            non_empty_sets.append(False)
            heatmap_data[:, subset_idx] = np.zeros(
                len(parent.feature_selection))

    heatmap_data = heatmap_data[parent.feature_selection]
    heatmap_data = heatmap_data[:, non_empty_sets]
    y_tick_labels = np.array(parent.col_names)[parent.feature_selection]
    x_tick_labels = np.array([subselection_var.get()
                              for subselection_var in parent.subset_names])
    x_tick_labels = x_tick_labels[non_empty_sets]
    if parent.initial_loop == False:
        parent.axs[subplot_idx].collections[-1].colorbar.remove()
    sns.heatmap(data=heatmap_data,
                ax=parent.axs[subplot_idx],
                yticklabels=y_tick_labels,
                xticklabels=x_tick_labels)
    plot_dict = {"type": "heatmap",
                 "subtype": "heatmap",
                 "subplot_idx": subplot_idx}
    parent.plot_dicts[subplot_idx] = plot_dict

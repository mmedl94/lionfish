
import numpy as np
import seaborn as sns


def launch_heatmap(parent, plot_object, subplot_idx):
    heatmap_data = np.empty(
        (len(parent.feature_selection), int(parent.n_subsets)))
    subset_sizes = np.empty(int(parent.n_subsets))
    cur_metric_var = parent.metric_vars[subplot_idx].get()

    if cur_metric_var not in ["Intra feature fraction",
                              "Intra cluster fraction",
                              "Total fraction"]:
        cur_metric_var = "Intra cluster fraction"
        parent.metric_vars[subplot_idx].set(cur_metric_var)

    # get matrix of summed counts
    non_empty_sets = []
    for subset_idx, subset in enumerate(parent.subselections):
        subset_sizes[subset_idx] = parent.data[subset].shape[0]
        if subset.shape[0] != 0:
            non_empty_sets.append(True)
            all_pos_feat_subset = np.sum(parent.data[subset], axis=0)
            heatmap_data[:, subset_idx] = all_pos_feat_subset
        else:
            non_empty_sets.append(False)
            heatmap_data[:, subset_idx] = np.zeros(
                len(parent.feature_selection))

    heatmap_data = heatmap_data[parent.feature_selection]
    heatmap_data = heatmap_data[:, non_empty_sets]
    subset_sizes = subset_sizes[non_empty_sets]

    if cur_metric_var == "Intra feature fraction":
        heatmap_data = heatmap_data/np.sum(heatmap_data, axis=1, keepdims=True)
    elif cur_metric_var == "Intra cluster fraction":
        heatmap_data = heatmap_data/subset_sizes
    elif cur_metric_var == "Total fraction":
        heatmap_data = heatmap_data/parent.data.shape[0]

    y_tick_labels = np.array(parent.feature_names)[parent.feature_selection]
    x_tick_labels = np.array([subselection_var.get()
                              for subselection_var in parent.subset_names])
    x_tick_labels = x_tick_labels[non_empty_sets]
    if parent.initial_loop == False:
        parent.axs[subplot_idx].collections[-1].colorbar.remove()

    sns.heatmap(data=heatmap_data,
                ax=parent.axs[subplot_idx],
                yticklabels=y_tick_labels,
                xticklabels=x_tick_labels)
    parent.axs[subplot_idx].set_xticks(parent.axs[subplot_idx].get_xticks(),
                                       parent.axs[subplot_idx].get_xticklabels(),
                                       rotation=25,
                                       ha="right")

    for label in parent.axs[subplot_idx].get_yticklabels():
        label.set_rotation(25)
        label.set_ha("right")
        label.set_rotation_mode("anchor")

    plot_dict = {"type": "heatmap",
                 "subtype": "heatmap",
                 "subplot_idx": subplot_idx}
    parent.plot_dicts[subplot_idx] = plot_dict

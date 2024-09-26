
import numpy as np


# Initialize data array
def launch_cat_clust_interface(parent, plot_object, subplot_idx):
    cat_clust_data = np.empty(
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
        subset_sizes[subset_idx] = max(parent.data[subset].shape[0], 1)
        if subset.shape[0] != 0:
            non_empty_sets.append(True)
            all_pos_feat_subset = np.sum(parent.data[subset], axis=0)
            cat_clust_data[:, subset_idx] = all_pos_feat_subset
        else:
            non_empty_sets.append(False)
            cat_clust_data[:, subset_idx] = np.ones(
                len(parent.feature_selection))

    cat_clust_data = cat_clust_data[parent.feature_selection]
    cat_clust_data = cat_clust_data[:, non_empty_sets]
    subset_sizes = subset_sizes[non_empty_sets]

    if cur_metric_var == "Intra feature fraction":
        cat_clust_data = cat_clust_data / \
            np.sum(cat_clust_data, axis=1, keepdims=True)
    elif cur_metric_var == "Intra cluster fraction":
        cat_clust_data = cat_clust_data/subset_sizes
    elif cur_metric_var == "Total fraction":
        cat_clust_data = cat_clust_data/parent.data.shape[0]

    # restructure data and sort descending based on metric values of subsets
    var_ids = np.repeat(np.arange(sum(parent.feature_selection)),
                        sum(non_empty_sets))
    cat_clust_data = cat_clust_data.flatten()
    clust_ids = np.arange(sum(non_empty_sets))
    clust_ids = np.tile(clust_ids, sum(non_empty_sets))
    # get current cluster selection
    for subselection_id, subselection_var in enumerate(parent.subselection_vars):
        if subselection_var.get() == 1:
            selected_cluster = subselection_id
    feature_selection_bool = np.repeat(
        parent.feature_selection, sum(non_empty_sets))
    if parent.initial_loop is False:
        parent.axs[subplot_idx].clear()
    x = cat_clust_data[feature_selection_bool]

    # Sort to display inter cluster max at the top
    sort_idx = np.arange(
        selected_cluster, x.shape[0], sum(non_empty_sets), dtype=int)
    ranked_vars = np.argsort(x[sort_idx])[::-1]
    sorting_helper = np.arange(x.shape[0])
    sorting_helper = sorting_helper.reshape(
        sort_idx.shape[0], int(sum(non_empty_sets)))
    sorting_helper = sorting_helper[ranked_vars].flatten()
    # flip var_ids so most important is on top
    var_ids = np.flip(var_ids)
    # Get coloration scheme
    fc = np.tile(np.array(parent.colors)[non_empty_sets],
                 (len(parent.feature_selection), 1))

    parent.axs[subplot_idx].scatter(
        x[sorting_helper],
        var_ids,
        c=fc[sorting_helper]
    )
    y_tick_labels = np.array(parent.feature_names)[parent.feature_selection]
    y_tick_labels = y_tick_labels[ranked_vars]
    # flip so that labels agree with var_ids
    y_tick_labels = np.flip(y_tick_labels)

    parent.axs[subplot_idx].set_yticks(
        np.arange(0, sum(parent.feature_selection)))
    parent.axs[subplot_idx].set_yticklabels(y_tick_labels,
                                            rotation=25,
                                            ha="right")
    parent.axs[subplot_idx].set_xlabel(cur_metric_var)
    if parent.subselections[selected_cluster].shape[0] == 0:
        fraction_of_total = 0
        subset_size = 0
    else:
        subset_size = parent.data[parent.subselections[selected_cluster]].shape[0]
        fraction_of_total = (
            subset_size/parent.data.shape[0])*100
    title = f"{subset_size} obsersvations - ({fraction_of_total:.2f}%)"
    parent.axs[subplot_idx].set_title(title)
    plot_dict = {"type": "cat_clust_interface",
                 "subtype": cur_metric_var,
                 "subplot_idx": subplot_idx,
                 "cat_clust_data": cat_clust_data}
    parent.plot_dicts[subplot_idx] = plot_dict

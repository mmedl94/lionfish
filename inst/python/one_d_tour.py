import numpy as np
from pytour_selectors import BarSelect, DraggableAnnotation1d


def launch_1d_tour(parent, plot_object, subplot_idx):

    if parent.initial_loop is True:
        parent.plot_dicts[subplot_idx]["reset_selection_check"] = False
        frame = 0
    else:
        frame = int(parent.frame_vars[subplot_idx].get())

    if frame >= plot_object["obj"].shape[-1]-1:
        frame = plot_object["obj"].shape[-1]-1
        parent.frame_vars[subplot_idx].set(str(frame))

    if "update_plot" in parent.plot_dicts[subplot_idx]:
        update_plot = parent.plot_dicts[subplot_idx]["update_plot"]
        parent.plot_dicts[subplot_idx]["update_plot"] = True
    else:
        update_plot = True

    if "reset_selection_check" not in parent.plot_dicts[subplot_idx]:
        parent.plot_dicts[subplot_idx]["reset_selection_check"] = False

    if update_plot is True:
        if parent.plot_dicts[subplot_idx]["reset_selection_check"] is False:
            proj = np.copy(plot_object["obj"][:, :, frame])
        else:
            proj = parent.plot_dicts[subplot_idx]["proj"]
            parent.plot_dicts[subplot_idx]["reset_selection_check"] = False
        data_subset = parent.data[:, parent.feature_selection]
        proj_subet = proj[parent.feature_selection][:, 0]
        proj_subet = proj_subet / \
            np.linalg.norm(proj_subet)
        x = np.matmul(data_subset, proj_subet)
        x = x/parent.half_range
        x = x-np.mean(x)
    else:
        proj = parent.plot_dicts[subplot_idx]["proj"]
        x = parent.plot_dicts[subplot_idx]["x"]

    parent.axs[subplot_idx].clear()
    # check if there are preselected points and update plot
    # recolor preselected points
    x_subselections = []
    for subselection in parent.subselections:
        if subselection.shape[0] != 0:
            x_subselections.append(x[subselection])
        else:
            x_subselections.append(np.array([]))

    hist = parent.axs[subplot_idx].hist(
        x_subselections,
        stacked=True,
        picker=True,
        color=parent.colors[:len(x_subselections)],
        bins=np.linspace(-1, 1, 26))
    y_lims = parent.axs[subplot_idx].get_ylim()
    parent.axs[subplot_idx].set_ylim(y_lims)

    if parent.initial_loop is True:
        parent.frame_vars[subplot_idx].set("0")
        parent.frame_textboxes[subplot_idx].configure(
            state="normal",
            fg_color="white")
        parent.fc = np.repeat(
            np.array(parent.colors[0])[:, np.newaxis], parent.n_pts, axis=1).T
        for idx, subset in enumerate(parent.subselections):
            if subset.shape[0] != 0:
                parent.fc[subset] = parent.colors[idx]
        plot_dict = {"type": "hist",
                     "subtype": "1d_tour",
                     "subplot_idx": subplot_idx,
                     "proj": proj}
        parent.plot_dicts[subplot_idx] = plot_dict
        bar_selector = BarSelect(parent=parent,
                                 subplot_idx=subplot_idx)
    else:
        parent.plot_dicts[subplot_idx]["draggable_annot"].disconnect()
        parent.plot_dicts[subplot_idx]["draggable_annot"].remove()
        plot_dict = {"type": "hist",
                     "subtype": "1d_tour",
                     "subplot_idx": subplot_idx,
                     "proj": proj}
        parent.plot_dicts[subplot_idx] = plot_dict
        bar_selector = BarSelect(parent=parent,
                                 subplot_idx=subplot_idx)

    parent.plot_dicts[subplot_idx]["selector"] = bar_selector
    draggable_arrows_1d = DraggableAnnotation1d(
        parent,
        subplot_idx,
        hist)
    parent.plot_dicts[subplot_idx]["draggable_annot"] = draggable_arrows_1d
    parent.axs[subplot_idx].set_xticks([])
    parent.axs[subplot_idx].set_yticks([])
    parent.axs[subplot_idx].set_xlim(-1, 1)

    parent.axs[subplot_idx].set_title(
        f"{parent.displayed_tour}\n" +
        "Press right key for next frame\n" +
        "Press left key for last frame")

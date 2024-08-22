import numpy as np
from helpers import gram_schmidt
from pytour_selectors import LassoSelect, DraggableAnnotation2d


def launch_2d_tour(parent, plot_object, subplot_idx):

    if parent.initial_loop is True:
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

        proj_subet = proj[parent.feature_selection]
        proj_subet[:, 0] = proj_subet[:, 0] / \
            np.linalg.norm(proj_subet[:, 0])
        proj_subet[:, 1] = gram_schmidt(
            proj_subet[:, 0], proj_subet[:, 1])
        proj_subet[:, 1] = proj_subet[:, 1] / \
            np.linalg.norm(proj_subet[:, 1])
        plot_data = parent.r.render_proj_inter(
            parent.data[:, parent.feature_selection], proj_subet,
            limits=parent.limits, half_range=parent.half_range)
        # Unpack tour data
        data_prj = plot_data["data_prj"]
        circle_prj = plot_data["circle"]
        x = data_prj.iloc[:, 0]
        y = data_prj.iloc[:, 1]
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
            scat = parent.axs[subplot_idx].scatter(
                x, y, animated=True)
            scat.set_facecolor(parent.fc)
            parent.original_fc = parent.fc.copy()
            parent.axs[subplot_idx].plot(circle_prj.iloc[:, 0],
                                         circle_prj.iloc[:, 1], color="grey")
        else:
            # clear old arrows and text
            for patch_idx, _ in enumerate(parent.axs[subplot_idx].patches):
                parent.axs[subplot_idx].patches[0].remove()
                parent.axs[subplot_idx].texts[0].remove()
            parent.plot_dicts[subplot_idx]["draggable_annot"].disconnect()
            parent.plot_dicts[subplot_idx]["selector"].disconnect()
            # update scatterplot
            parent.plot_dicts[subplot_idx]["scat"].set_offsets(
                np.array([x, y]).T)
            scat = parent.axs[subplot_idx].collections[0]
            scat.set_facecolors(parent.fc)
        parent.axs[subplot_idx].set_xlim(-parent.limits *
                                         1.1, parent.limits*1.1)
        parent.axs[subplot_idx].set_ylim(-parent.limits *
                                         1.1, parent.limits*1.1)
        parent.axs[subplot_idx].set_xticks([])
        parent.axs[subplot_idx].set_yticks([])
        parent.axs[subplot_idx].set_aspect("equal")
        plot_dict = {"type": "scatter",
                     "subtype": "2d_tour",
                     "subplot_idx": subplot_idx,
                     "scat": scat,
                     "proj": proj,
                     "update_plot": True
                     }
        parent.plot_dicts[subplot_idx] = plot_dict
        # start Lasso selector
        selector = LassoSelect(
            parent=parent,
            subplot_idx=subplot_idx
        )
        parent.plot_dicts[subplot_idx]["selector"] = selector
        plot_dict["draggable_annot"] = DraggableAnnotation2d(
            parent,
            subplot_idx
        )
        n_frames = plot_object["obj"].shape[-1]-1

        parent.axs[subplot_idx].set_title(
            f"{parent.displayed_tour}\n" +
            "Press right key for next frame\n" +
            "Press left key for last frame")
    else:
        parent.plot_dicts[subplot_idx]["scat"].set_facecolors(parent.fc)
        parent.plot_dicts[subplot_idx]["selector"].pause_var = parent.pause_var

import numpy as np

from helpers import gram_schmidt
from pytour_selectors import DraggableAnnotation1d, DraggableAnnotation2d, LassoSelect


def feature_checkbox_event(self, feature_idx):
    feature_selection = [state.get()
                         for state in self.feature_selection_vars]
    if sum(feature_selection) <= 1:
        self.feature_selection_vars[feature_idx].set(1)
    self.feature_selection = [state.get()
                              for state in self.feature_selection_vars]
    self.feature_selection = np.bool_(self.feature_selection)

    for subplot_idx, plot_dict in enumerate(self.plot_dicts):
        if plot_dict["subtype"] == "2d_tour":
            data_subset = self.data[:, self.feature_selection]
            proj_subet = plot_dict["proj"][self.feature_selection]

            proj_subet[:, 0] = proj_subet[:, 0] / \
                np.linalg.norm(proj_subet[:, 0])
            proj_subet[:, 1] = gram_schmidt(
                proj_subet[:, 0], proj_subet[:, 1])
            proj_subet[:, 1] = proj_subet[:, 1] / \
                np.linalg.norm(proj_subet[:, 1])

            plot_data = self.r.render_proj_inter(
                data_subset, proj_subet, limits=self.limits, half_range=self.half_range)
            # Unpack tour data

            data_prj = np.matmul(self.data[:, self.feature_selection],
                                 proj_subet)/self.half_range
            circle_prj = plot_data["circle"]
            x = data_prj[:, 0]
            y = data_prj[:, 1]

            old_title = plot_dict["ax"].get_title()

            # clear old scatterplot
            plot_dict["ax"].clear()
            # Make new scatterplot
            scat = plot_dict["ax"].scatter(x, y)
            scat = plot_dict["ax"].collections[0]
            scat.set_facecolors(plot_dict["fc"])

            plot_dict["ax"].set_xlim(-self.limits*1.1, self.limits*1.1)
            plot_dict["ax"].set_ylim(-self.limits*1.1, self.limits*1.1)
            plot_dict["ax"].set_box_aspect(aspect=1)

            plot_dict["ax"].plot(circle_prj.iloc[:, 0],
                                 circle_prj.iloc[:, 1], color="gray")

            self.axs[subplot_idx].set_title(old_title)

            self.plot_dicts[subplot_idx]["ax"] = plot_dict["ax"]

            # start Lasso selector
            self.selectors[subplot_idx] = LassoSelect(
                plot_dicts=self.plot_dicts,
                subplot_idx=subplot_idx,
                colors=self.colors,
                n_pts=self.n_pts,
                alpha_other=self.alpha_other)

            plot_dict["draggable_annot"] = DraggableAnnotation2d(
                self.data,
                plot_dict["proj"],
                plot_dict["ax"],
                scat,
                self.half_range,
                self.feature_selection,
                self.col_names)

            self.plot_dicts[subplot_idx]["ax"].figure.canvas.draw_idle()

        if plot_dict["subtype"] == "1d_tour":
            data_subset = self.data[:, self.feature_selection]
            proj_subet = plot_dict["proj"][self.feature_selection]
            proj_subet = proj_subet/np.linalg.norm(proj_subet)

            x = np.matmul(data_subset, proj_subet)[:, 0]
            x = x/self.half_range
            title = plot_dict["ax"].get_title()
            plot_dict["ax"].clear()

            # check if there are preselected points and update plot
            x_subselections = []
            for subselection in self.plot_dicts[0]["subselections"]:
                if subselection.shape[0] != 0:
                    x_subselections.append(x[subselection])
                else:
                    x_subselections.append(np.array([]))
            plot_dict["ax"].clear()

            hist = plot_dict["ax"].hist(
                x_subselections,
                stacked=True,
                picker=True,
                color=self.colors[:len(x_subselections)])

            self.plot_dicts[subplot_idx]["arrows"].remove()
            draggable_arrows_1d = DraggableAnnotation1d(
                self.data,
                self.plot_dicts,
                subplot_idx,
                hist,
                self.half_range,
                self.feature_selection,
                self.colors,
                self.col_names)
            self.plot_dicts[subplot_idx]["arrows"] = draggable_arrows_1d
            plot_dict["ax"].set_xlim(-1, 1)
            plot_dict["ax"].set_title(title)
            plot_dict["data"] = self.data

            self.plot_dicts[subplot_idx] = plot_dict
            self.plot_dicts[subplot_idx]["ax"].figure.canvas.draw_idle()


def subselection_checkbox_event(self, subselection_idx):
    for i, _ in enumerate(self.subselection_vars):
        if i == subselection_idx:
            self.subselection_vars[i].set(1)
        else:
            self.subselection_vars[i].set(0)

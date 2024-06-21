import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from functools import partial

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import customtkinter as ctk

from helpers import gram_schmidt
from pytour_selectors import LassoSelect, DraggableAnnotation1d, DraggableAnnotation2d, BarSelect


class InteractiveTourInterface(ctk.CTk):
    def __init__(self, data, col_names, plot_objects, half_range, n_max_cols):
        super().__init__()
        self.title("Interactive tourr")
        self.data = data
        self.col_names = col_names
        self.half_range = half_range

        if not isinstance(plot_objects, list):
            plot_objects = [plot_objects]

        # if len(plot_objects[0]) == 2:
        #    [plot_objects] = np.expand_dims(plot_objects[0], axis=2)

        if half_range is None:
            print("Using adaptive half_range")
        else:
            print(f"Using half_range of {half_range}")

        limits = 1
        self.alpha_other = 0.3
        n_pts = self.data.shape[0]
        # Initialize self.obs_idx with all obs
        self.obs_idx_ = np.arange(0, self.data.shape[0])
        if len(plot_objects) == 1:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            axs = [axs]
        else:
            n_plots = len(plot_objects)
            if n_max_cols is None:
                n_max_cols = 3
            n_cols = int(min(n_max_cols, n_plots))
            n_rows = int((n_plots + n_cols - 1) // n_cols)
            if n_rows == 1:
                fig, axs = plt.subplots(
                    n_rows, n_cols, figsize=(15, 15/n_cols), layout="compressed")
            else:
                fig, axs = plt.subplots(
                    n_rows, n_cols, figsize=(15, 10), layout="compressed")
            axs = axs.flatten()
            for i in range(n_plots, len(axs)):
                fig.delaxes(axs[i])
            axs = axs[:n_plots]

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=1, sticky="n")

        toolbar = NavigationToolbar2Tk(canvas, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, columnspan=2, sticky="w")

        feature_selection_frame = ctk.CTkFrame(self)
        feature_selection_frame.grid(row=0, column=0, sticky="n")

        def checkbox_event(feature_idx):
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

                    plot_data = r.render_proj_inter(
                        data_subset, proj_subet, limits=limits, half_range=half_range)
                    # Unpack tour data

                    data_prj = np.matmul(self.data[:, self.feature_selection],
                                         proj_subet)/half_range
                    circle_prj = plot_data["circle"]
                    x = data_prj[:, 0]
                    y = data_prj[:, 1]

                    old_title = plot_dict["ax"].get_title()

                    # clear old scatterplot
                    plot_dict["ax"].clear()
                    # Make new scatterplot
                    scat = plot_dict["ax"].scatter(x, y)
                    scat = plot_dict["ax"].collections[0]
                    plot_dict["ax"].set_xlim(-limits*1.1, limits*1.1)
                    plot_dict["ax"].set_ylim(-limits*1.1, limits*1.1)
                    plot_dict["ax"].set_box_aspect(aspect=1)

                    plot_dict["ax"].plot(circle_prj.iloc[:, 0],
                                         circle_prj.iloc[:, 1], color="gray")

                    axs[subplot_idx].set_title(old_title)

                    # Recolor preselected points
                    if self.last_selection[0] is not False:
                        fc = scat.get_facecolors()
                        fc = np.tile(fc, (n_pts, 1))
                        fc[:, -1] = self.alpha_other
                        fc[self.last_selection, -1] = 1
                        scat.set_facecolors(fc)

                    self.plot_dicts[subplot_idx]["ax"] = plot_dict["ax"]

                    # start Lasso selector
                    self.selectors[subplot_idx] = LassoSelect(
                        plot_dicts=self.plot_dicts,
                        subplot_idx=subplot_idx,
                        alpha_other=self.alpha_other,
                        last_selection=self.last_selection)

                    plot_dict["draggable_annot"] = DraggableAnnotation2d(
                        self.data,
                        plot_dict["proj"],
                        plot_dict["ax"],
                        scat,
                        half_range,
                        self.feature_selection,
                        col_names)

                    self.plot_dicts[subplot_idx]["ax"].figure.canvas.draw_idle()

                if plot_dict["subtype"] == "1d_tour":
                    data_subset = self.data[:, self.feature_selection]
                    proj_subet = plot_dict["proj"][self.feature_selection]
                    proj_subet = proj_subet/np.linalg.norm(proj_subet)

                    x = np.matmul(data_subset, proj_subet)[:, 0]
                    x = x/half_range
                    title = plot_dict["ax"].get_title()
                    plot_dict["ax"].clear()

                    # check if there are preselected points and update plot
                    if self.last_selection[0] is not False:
                        # recolor preselected points
                        selected_obs = x[self.last_selection[0]]
                        other_obs = np.delete(x, self.last_selection[0])

                        fc_sel = self.plot_dicts[subplot_idx]["fc"]
                        fc_sel[-1] = 1
                        fc_not_sel = fc_sel.copy()
                        fc_not_sel[-1] = self.alpha_other
                        color_map = [fc_sel, fc_not_sel]
                        hist = plot_dict["ax"].hist(
                            [selected_obs, other_obs],
                            stacked=True,
                            picker=True,
                            color=color_map)
                    else:
                        hist = plot_dict["ax"].hist(x, picker=True)
                        fc_sel = list(hist[2][0].get_facecolor())
                    self.plot_dicts[subplot_idx]["arrows"].remove()
                    draggable_arrows_1d = DraggableAnnotation1d(
                        self.data,
                        self.plot_dicts,
                        subplot_idx,
                        hist,
                        half_range,
                        self.feature_selection,
                        self.last_selection,
                        col_names)
                    self.plot_dicts[subplot_idx]["arrows"] = draggable_arrows_1d
                    plot_dict["ax"].set_xlim(-1, 1)
                    plot_dict["ax"].set_title(title)
                    plot_dict["data"] = self.data

                    self.plot_dicts[subplot_idx] = plot_dict
                    self.plot_dicts[subplot_idx]["ax"].figure.canvas.draw_idle()

        self.feature_selection_vars = []
        self.feature_selection = []
        for feature_idx, feature in enumerate(col_names):
            check_var = tk.IntVar(self, 1)
            self.feature_selection_vars.append(check_var)
            self.feature_selection.append(1)
            checkbox = ctk.CTkCheckBox(master=feature_selection_frame,
                                       text=feature,
                                       command=partial(
                                           checkbox_event, feature_idx),
                                       variable=check_var,
                                       onvalue=1,
                                       offvalue=0)
            checkbox.grid(row=feature_idx, column=0, pady=3)
        self.feature_selection = np.bool_(self.feature_selection)

        # Get max number of frames
        self.n_frames = 0
        for plot_object in plot_objects:
            if plot_object["type"] == "hist":
                pass
            elif plot_object["type"] == "scatter":
                pass
            elif plot_object["obj"].shape[-1] > self.n_frames:
                self.n_frames = plot_object["obj"].shape[-1]
        self.frame = 0

        # resolve while loop in case of window closing
        def cleanup():
            self.frame = self.n_frames
            self.pause_var.set(1)
        self.protocol("WM_DELETE_WINDOW", cleanup)

        def accept(event):
            if event.key == "right" or event.key == "left":
                self.initial_loop = False
                fig.canvas.draw()
                if event.key == "right":
                    self.frame += 1
                if event.key == "left" and self.frame > 0:
                    self.frame -= 1
                self.pause_var.set(1)

        self.plot_dicts = [i for i, _ in enumerate(plot_objects)]
        self.last_selection = [False]
        self.initial_loop = True

        while self.frame < self.n_frames:
            self.selectors = []
            for subplot_idx, plot_object in enumerate(plot_objects):
                frame = self.frame

                if plot_object["type"] == "2d_tour":
                    if frame >= plot_object["obj"].shape[-1]-1:
                        frame = plot_object["obj"].shape[-1]-1
                    # get tour data
                    proj = np.copy(
                        plot_object["obj"][:, :, frame])
                    proj_subet = proj[self.feature_selection]

                    proj_subet[:, 0] = proj_subet[:, 0] / \
                        np.linalg.norm(proj_subet[:, 0])
                    proj_subet[:, 1] = gram_schmidt(
                        proj_subet[:, 0], proj_subet[:, 1])
                    proj_subet[:, 1] = proj_subet[:, 1] / \
                        np.linalg.norm(proj_subet[:, 1])

                    plot_data = r.render_proj_inter(
                        self.data[:, self.feature_selection], proj_subet, limits=limits, half_range=half_range)
                    # Unpack tour data
                    data_prj = plot_data["data_prj"]
                    circle_prj = plot_data["circle"]
                    x = data_prj.iloc[:, 0]
                    y = data_prj.iloc[:, 1]

                    # clear old scatterplot
                    axs[subplot_idx].clear()
                    # Make new scatterplot
                    scat = axs[subplot_idx].scatter(x, y)
                    scat = axs[subplot_idx].collections[0]
                    axs[subplot_idx].set_xlim(-limits * 1.1, limits*1.1)
                    axs[subplot_idx].set_ylim(-limits * 1.1, limits*1.1)
                    axs[subplot_idx].set_box_aspect(aspect=1)

                    # Recolor preselected points
                    if self.last_selection[0] is not False:
                        fc = scat.get_facecolors()
                        fc = np.tile(fc, (n_pts, 1))
                        fc[:, -1] = self.alpha_other
                        fc[self.last_selection, -1] = 1
                        scat.set_facecolors(fc)

                    plot_dict = {"type": "scatter",
                                 "subtype": "2d_tour",
                                 "subplot_idx": subplot_idx,
                                 "ax": axs[subplot_idx],
                                 "feature_selection": self.feature_selection,
                                 "proj": proj
                                 }
                    self.plot_dicts[subplot_idx] = plot_dict

                    # start Lasso selector
                    selector = LassoSelect(
                        plot_dicts=self.plot_dicts,
                        subplot_idx=subplot_idx,
                        alpha_other=self.alpha_other,
                        last_selection=self.last_selection)
                    self.selectors.append(selector)

                    plot_dict["draggable_annot"] = DraggableAnnotation2d(
                        self.data,
                        self.plot_dicts[subplot_idx]["proj"],
                        axs[subplot_idx],
                        scat,
                        half_range,
                        self.feature_selection,
                        col_names)

                    axs[subplot_idx].plot(circle_prj.iloc[:, 0],
                                          circle_prj.iloc[:, 1], color="grey")
                    n_frames = plot_object["obj"].shape[-1]-1
                    axs[subplot_idx].set_title(f"Frame {frame} out of {n_frames}" +
                                               f"\nPress right key for next frame" +
                                               f"\nPress left key for last frame")

                if plot_object["type"] == "1d_tour":
                    if frame >= plot_object["obj"].shape[-1]-1:
                        frame = plot_object["obj"].shape[-1]-1

                    proj = np.copy(
                        plot_object["obj"][:, :, frame])

                    data_subset = self.data[:, self.feature_selection]
                    proj_subet = proj[self.feature_selection][:, 0]
                    proj_subet = proj_subet / \
                        np.linalg.norm(proj_subet)
                    x = np.matmul(data_subset, proj_subet)
                    x = x/half_range

                    axs[subplot_idx].clear()

                    # check if there are preselected points and update plot
                    if self.last_selection[0] is not False:
                        # recolor preselected points
                        selected_obs = x[self.last_selection[0]]
                        other_obs = np.delete(x, self.last_selection[0])
                        fc_sel = self.plot_dicts[subplot_idx]["fc"]
                        fc_sel[-1] = 1
                        fc_not_sel = fc_sel.copy()
                        fc_not_sel[-1] = self.alpha_other
                        color_map = [fc_sel, fc_not_sel]
                        hist = axs[subplot_idx].hist(
                            [selected_obs, other_obs],
                            stacked=True,
                            picker=True,
                            color=color_map)
                        y_lims = axs[subplot_idx].get_ylim()
                        axs[subplot_idx].set_ylim(y_lims)
                    else:
                        hist = axs[subplot_idx].hist(x, picker=True)
                        fc_sel = list(hist[2][0].get_facecolor())

                    if self.initial_loop is True:
                        plot_dict = {"type": "hist",
                                     "subtype": "1d_tour",
                                     "subplot_idx": subplot_idx,
                                     "ax": axs[subplot_idx],
                                     "data": self.data,
                                     "feature_selection": self.feature_selection,
                                     "half_range": half_range,
                                     "fc": fc_sel,
                                     "proj": proj}
                        self.plot_dicts[subplot_idx] = plot_dict
                        bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                 subplot_idx=subplot_idx,
                                                 feature_selection=self.feature_selection,
                                                 half_range=self.half_range,
                                                 alpha_other=self.alpha_other,
                                                 last_selection=self.last_selection)
                    else:
                        self.plot_dicts[subplot_idx]["arrows"].remove()
                        plot_dict = {"type": "hist",
                                     "subtype": "1d_tour",
                                     "subplot_idx": subplot_idx,
                                     "ax": axs[subplot_idx],
                                     "data": self.data,
                                     "feature_selection": self.feature_selection,
                                     "half_range": half_range,
                                     "fc": fc_sel,
                                     "proj": proj,
                                     "selector": self.plot_dicts[subplot_idx]["selector"]}
                        self.plot_dicts[subplot_idx] = plot_dict
                        self.plot_dicts[subplot_idx]["selector"].disconnect()
                        bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                 subplot_idx=subplot_idx,
                                                 feature_selection=self.feature_selection,
                                                 half_range=self.half_range,
                                                 alpha_other=self.alpha_other,
                                                 last_selection=self.last_selection)
                    self.plot_dicts[subplot_idx]["selector"] = bar_selector

                    draggable_arrows_1d = DraggableAnnotation1d(
                        self.data,
                        self.plot_dicts,
                        subplot_idx,
                        hist,
                        half_range,
                        self.feature_selection,
                        self.last_selection,
                        col_names)

                    self.plot_dicts[subplot_idx]["arrows"] = draggable_arrows_1d

                    n_frames = plot_object["obj"].shape[-1]-1
                    axs[subplot_idx].set_xlim(-1, 1)
                    axs[subplot_idx].set_title(f"Frame {frame} out of {n_frames}" +
                                               f"\nPress right key for next frame" +
                                               f"\nPress left key for last frame")

                if plot_object["type"] == "scatter":
                    # get data
                    col_index_x = col_names.index(plot_object["obj"][0])
                    col_index_y = col_names.index(plot_object["obj"][1])
                    x = self.data[:, col_index_x]
                    y = self.data[:, col_index_y]

                    # clear old scatterplot
                    axs[subplot_idx].clear()
                    # Make new scatterplot
                    scat = axs[subplot_idx].scatter(x, y)
                    scat = axs[subplot_idx].collections[0]
                    x_lims = axs[subplot_idx].get_xlim()
                    y_lims = axs[subplot_idx].get_ylim()

                    axs[subplot_idx].set_xlim(x_lims)
                    axs[subplot_idx].set_ylim(y_lims)
                    axs[subplot_idx].set_box_aspect(aspect=1)

                    # Recolor preselected points
                    if self.last_selection[0] is not False:
                        fc = scat.get_facecolors()
                        fc = np.tile(fc, (n_pts, 1))
                        fc[:, -1] = self.alpha_other
                        fc[self.last_selection, -1] = 1
                        scat.set_facecolors(fc)

                    plot_dict = {"type": "scatter",
                                 "subtype": "scatter",
                                 "subplot_idx": subplot_idx,
                                 "ax": axs[subplot_idx]
                                 }
                    self.plot_dicts[subplot_idx] = plot_dict
                    # start Lasso selector
                    selector = LassoSelect(
                        plot_dicts=self.plot_dicts,
                        subplot_idx=subplot_idx,
                        alpha_other=self.alpha_other,
                        last_selection=self.last_selection)
                    self.selectors.append(selector)
                    x_name = plot_object["obj"][0]
                    y_name = plot_object["obj"][1]
                    axs[subplot_idx].set_xlabel(x_name)
                    axs[subplot_idx].set_ylabel(y_name)
                    axs[subplot_idx].set_title(
                        f"Scatterplot of variables {x_name} and {y_name}")

                elif plot_object["type"] == "hist":
                    if plot_object["obj"] in col_names:
                        col_index = col_names.index(plot_object["obj"])
                        x = self.data[:, col_index]
                        # clear old histogram
                        axs[subplot_idx].clear()

                        if self.last_selection[0] is not False:
                            # recolor preselected points
                            selected_obs = x[self.last_selection][0]
                            other_obs = np.delete(
                                x, self.last_selection)
                            fc_sel = self.plot_dicts[subplot_idx]["fc"]
                            fc_sel[-1] = 1
                            fc_not_sel = fc_sel.copy()
                            fc_not_sel[-1] = self.alpha_other

                            color_map = [fc_sel, fc_not_sel]
                            hist = axs[subplot_idx].hist(
                                [selected_obs, other_obs],
                                stacked=True,
                                picker=True,
                                color=color_map)
                            y_lims = axs[subplot_idx].get_ylim()
                            axs[subplot_idx].set_ylim(y_lims)
                        else:
                            hist = axs[subplot_idx].hist(x, picker=True)
                            axs[subplot_idx].set_box_aspect(aspect=1)
                            fc_sel = list(hist[2][0].get_facecolor())

                        axs[subplot_idx].set_box_aspect(aspect=1)
                        hist_variable_name = plot_object["obj"]
                        axs[subplot_idx].set_xlabel(hist_variable_name)
                        axs[subplot_idx].set_title(
                            f"Histogram of variable {hist_variable_name}")

                        if self.initial_loop is True:
                            plot_dict = {"type": "hist",
                                         "subtype": "hist",
                                         "subplot_idx": subplot_idx,
                                         "ax": axs[subplot_idx],
                                         "data": self.data,
                                         "hist_feature": col_index,
                                         "feature_selection": self.feature_selection,
                                         "half_range": half_range,
                                         "fc": fc_sel,
                                         "proj": proj}
                            self.plot_dicts[subplot_idx] = plot_dict
                            bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                     subplot_idx=subplot_idx,
                                                     feature_selection=self.feature_selection,
                                                     half_range=self.half_range,
                                                     alpha_other=self.alpha_other,
                                                     last_selection=self.last_selection)
                            self.plot_dicts[subplot_idx]["selector"] = bar_selector
                        else:
                            plot_dict = {"type": "hist",
                                         "subtype": "hist",
                                         "subplot_idx": subplot_idx,
                                         "ax": axs[subplot_idx],
                                         "data": self.data,
                                         "hist_feature": col_index,
                                         "feature_selection": self.feature_selection,
                                         "half_range": half_range,
                                         "fc": fc_sel,
                                         "proj": proj,
                                         "selector": self.plot_dicts[subplot_idx]["selector"]}
                            self.plot_dicts[subplot_idx] = plot_dict
                            self.plot_dicts[subplot_idx]["selector"].disconnect(
                            )
                            bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                     subplot_idx=subplot_idx,
                                                     feature_selection=self.feature_selection,
                                                     half_range=self.half_range,
                                                     alpha_other=self.alpha_other,
                                                     last_selection=self.last_selection)
                            self.plot_dicts[subplot_idx]["selector"] = bar_selector
                    else:
                        print("Column not found")

            fig.canvas.draw()
            self.pause_var = tk.StringVar()
            fig.canvas.mpl_connect("key_press_event", accept)
            self.wait_variable(self.pause_var)

        self.destroy()


def interactive_tour(data, col_names, plot_objects, half_range=None, n_max_cols=None):
    """Launch InteractiveTourInterface object"""
    app = InteractiveTourInterface(data,
                                   col_names,
                                   plot_objects,
                                   half_range,
                                   n_max_cols)
    app.mainloop()

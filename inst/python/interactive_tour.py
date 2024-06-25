import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from functools import partial

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import customtkinter as ctk

from helpers import gram_schmidt
from pytour_selectors import LassoSelect, DraggableAnnotation1d, DraggableAnnotation2d, BarSelect
from checkbox_events import feature_checkbox_event, subselection_checkbox_event


def interactive_tour(data, col_names, plot_objects, half_range=None, n_max_cols=None, preselection=None, n_subsets=3):
    """Launch InteractiveTourInterface object"""
    app = InteractiveTourInterface(data,
                                   col_names,
                                   plot_objects,
                                   half_range,
                                   n_max_cols,
                                   preselection,
                                   n_subsets)
    app.mainloop()


class InteractiveTourInterface(ctk.CTk):
    def __init__(self, data, col_names, plot_objects, half_range, n_max_cols,
                 preselection, n_subsets):
        super().__init__()
        self.title("Interactive tourr")
        self.data = data
        self.col_names = col_names
        self.half_range = half_range
        self.r = r
        self.n_subsets = int(n_subsets)
        self.preselection = np.array(preselection, dtype=int)-1
        self.colors = matplotlib.colormaps["tab10"].colors

        if not isinstance(plot_objects, list):
            plot_objects = [plot_objects]

        # if len(plot_objects[0]) == 2:
        #    [plot_objects] = np.expand_dims(plot_objects[0], axis=2)

        if half_range is None:
            print("Using adaptive half_range")
        else:
            print(f"Using half_range of {half_range}")

        self.limits = 1
        self.alpha_other = 0.3
        self.n_pts = self.data.shape[0]
        # Initialize self.obs_idx with all obs
        self.obs_idx_ = np.arange(0, self.data.shape[0])
        if len(plot_objects) == 1:
            fig, self.axs = plt.subplots(1, 1, figsize=(10, 10))
            self.axs = [self.axs]
        else:
            n_plots = len(plot_objects)
            if n_max_cols is None:
                n_max_cols = 3
            n_cols = int(min(n_max_cols, n_plots))
            n_rows = int((n_plots + n_cols - 1) // n_cols)
            if n_rows == 1:
                fig, self.axs = plt.subplots(
                    n_rows, n_cols, figsize=(15, 15/n_cols), layout="compressed")
            else:
                fig, self.axs = plt.subplots(
                    n_rows, n_cols, figsize=(15, 10), layout="compressed")
            self.axs = self.axs.flatten()
            for i in range(n_plots, len(self.axs)):
                fig.delaxes(self.axs[i])
            self.axs = self.axs[:n_plots]

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=1, sticky="n")

        toolbar = NavigationToolbar2Tk(canvas, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, columnspan=2, sticky="w")

        sidebar = ctk.CTkFrame(self)
        sidebar.grid(row=0, column=0, sticky="n")

        feature_selection_frame = ctk.CTkFrame(sidebar)
        feature_selection_frame.grid(row=0, column=0, sticky="n")

        self.feature_selection_vars = []
        self.feature_selection = []
        for feature_idx, feature in enumerate(col_names):
            check_var = tk.IntVar(self, 1)
            self.feature_selection_vars.append(check_var)
            self.feature_selection.append(1)
            checkbox = ctk.CTkCheckBox(master=feature_selection_frame,
                                       text=feature,
                                       command=partial(
                                           feature_checkbox_event, self, feature_idx),
                                       variable=check_var,
                                       onvalue=1,
                                       offvalue=0)
            checkbox.grid(row=feature_idx, column=0, pady=3)
        self.feature_selection = np.bool_(self.feature_selection)

        subselection_frame = ctk.CTkFrame(sidebar)
        subselection_frame.grid(row=1, column=0, sticky="n")

        self.subselection_vars = []
        self.subselections = []
        for subselection_idx in range(self.n_subsets):
            if preselection is not None:
                if subselection_idx == 0:
                    check_var = tk.IntVar(self, 1)
                    preselection_ind = np.where(
                        self.preselection == subselection_idx)
                    self.subselections.append(preselection_ind[0])
                elif subselection_idx < len(set(self.preselection)):
                    check_var = tk.IntVar(self, 0)
                    preselection_ind = np.where(
                        self.preselection == subselection_idx)
                    self.subselections.append(preselection_ind[0])
                else:
                    check_var = tk.IntVar(self, 0)
                    self.subselections.append(np.array([]))
            else:
                if subselection_idx == 0:
                    check_var = tk.IntVar(self, 1)
                    self.subselections.append(np.arange(self.n_pts))
                else:
                    check_var = tk.IntVar(self, 0)
                    self.subselections.append(np.array([]))
            self.subselection_vars.append(check_var)
            checkbox = ctk.CTkCheckBox(master=subselection_frame,
                                       text=f"Subselection {subselection_idx+1}",
                                       command=partial(
                                           subselection_checkbox_event, self, subselection_idx),
                                       variable=check_var,
                                       onvalue=1,
                                       offvalue=0)
            checkbox.grid(row=subselection_idx, column=0, pady=3)

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
                    plot_data = self.r.render_proj_inter(
                        self.data[:, self.feature_selection], proj_subet,
                        limits=self.limits, half_range=half_range)
                    # Unpack tour data
                    data_prj = plot_data["data_prj"]
                    circle_prj = plot_data["circle"]
                    x = data_prj.iloc[:, 0]
                    y = data_prj.iloc[:, 1]

                    if self.initial_loop is True:
                        self.fc = np.repeat(
                            np.array(self.colors[0])[:, np.newaxis], self.n_pts, axis=1).T
                        for idx, subset in enumerate(self.subselections):
                            if subset.shape[0] != 0:
                                self.fc[subset] = self.colors[idx]
                        scat = self.axs[subplot_idx].scatter(x, y)
                        scat.set_facecolor(self.fc)
                    else:
                        # clear old scatterplot
                        self.axs[subplot_idx].clear()
                        # Make new scatterplot
                        scat = self.axs[subplot_idx].scatter(x, y)
                        scat = self.axs[subplot_idx].collections[0]
                        scat.set_facecolors(
                            self.plot_dicts[subplot_idx]["fc"])
                        self.fc = self.plot_dicts[subplot_idx]["fc"]

                    self.axs[subplot_idx].set_xlim(-self.limits *
                                                   1.1, self.limits*1.1)
                    self.axs[subplot_idx].set_ylim(-self.limits *
                                                   1.1, self.limits*1.1)
                    self.axs[subplot_idx].set_box_aspect(aspect=1)

                    plot_dict = {"type": "scatter",
                                 "subtype": "2d_tour",
                                 "subplot_idx": subplot_idx,
                                 "ax": self.axs[subplot_idx],
                                 "feature_selection": self.feature_selection,
                                 "subselection_vars": self.subselection_vars,
                                 "subselections": self.subselections,
                                 "fc": self.fc,
                                 "proj": proj
                                 }
                    self.plot_dicts[subplot_idx] = plot_dict

                    # start Lasso selector
                    selector = LassoSelect(
                        plot_dicts=self.plot_dicts,
                        subplot_idx=subplot_idx,
                        colors=self.colors,
                        n_pts=self.n_pts,
                        alpha_other=self.alpha_other)
                    self.selectors.append(selector)

                    plot_dict["draggable_annot"] = DraggableAnnotation2d(
                        self.data,
                        self.plot_dicts[subplot_idx]["proj"],
                        self.axs[subplot_idx],
                        scat,
                        half_range,
                        self.feature_selection,
                        col_names)

                    self.axs[subplot_idx].plot(circle_prj.iloc[:, 0],
                                               circle_prj.iloc[:, 1], color="grey")
                    n_frames = plot_object["obj"].shape[-1]-1
                    self.axs[subplot_idx].set_title(f"Frame {frame} out of {n_frames}" +
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

                    self.axs[subplot_idx].clear()

                    # check if there are preselected points and update plot
                    # recolor preselected points
                    x_subselections = []
                    for subselection in self.plot_dicts[0]["subselections"]:
                        if subselection.shape[0] != 0:
                            x_subselections.append(x[subselection])
                        else:
                            x_subselections.append(np.array([]))
                    hist = self.axs[subplot_idx].hist(
                        x_subselections,
                        stacked=True,
                        picker=True,
                        color=self.colors[:len(x_subselections)])
                    y_lims = self.axs[subplot_idx].get_ylim()
                    self.axs[subplot_idx].set_ylim(y_lims)

                    if self.initial_loop is True:
                        self.fc = np.repeat(
                            np.array(self.colors[0])[:, np.newaxis], self.n_pts, axis=1).T
                        for idx, subset in enumerate(self.subselections):
                            if subset.shape[0] != 0:
                                self.fc[subset] = self.colors[idx]
                        plot_dict = {"type": "hist",
                                     "subtype": "1d_tour",
                                     "subplot_idx": subplot_idx,
                                     "ax": self.axs[subplot_idx],
                                     "data": self.data,
                                     "feature_selection": self.feature_selection,
                                     "subselection_vars": self.subselection_vars,
                                     "subselections": self.subselections,
                                     "half_range": half_range,
                                     "fc": self.fc,
                                     "proj": proj}
                        self.plot_dicts[subplot_idx] = plot_dict
                        bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                 subplot_idx=subplot_idx,
                                                 feature_selection=self.feature_selection,
                                                 colors=self.colors,
                                                 half_range=self.half_range,
                                                 alpha_other=self.alpha_other)
                    else:
                        self.plot_dicts[subplot_idx]["arrows"].remove()
                        plot_dict = {"type": "hist",
                                     "subtype": "1d_tour",
                                     "subplot_idx": subplot_idx,
                                     "ax": self.axs[subplot_idx],
                                     "data": self.data,
                                     "feature_selection": self.feature_selection,
                                     "subselection_vars": self.subselection_vars,
                                     "subselections": self.subselections,
                                     "half_range": half_range,
                                     "proj": proj,
                                     "fc": self.fc,
                                     "selector": self.plot_dicts[subplot_idx]["selector"]}
                        self.plot_dicts[subplot_idx] = plot_dict
                        self.plot_dicts[subplot_idx]["selector"].disconnect()
                        bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                 subplot_idx=subplot_idx,
                                                 feature_selection=self.feature_selection,
                                                 colors=self.colors,
                                                 half_range=self.half_range,
                                                 alpha_other=self.alpha_other)
                    self.plot_dicts[subplot_idx]["selector"] = bar_selector

                    draggable_arrows_1d = DraggableAnnotation1d(
                        self.data,
                        self.plot_dicts,
                        subplot_idx,
                        hist,
                        half_range,
                        self.feature_selection,
                        self.colors,
                        col_names)

                    self.plot_dicts[subplot_idx]["arrows"] = draggable_arrows_1d

                    n_frames = plot_object["obj"].shape[-1]-1
                    self.axs[subplot_idx].set_xlim(-1, 1)
                    self.axs[subplot_idx].set_title(f"Frame {frame} out of {n_frames}" +
                                                    f"\nPress right key for next frame" +
                                                    f"\nPress left key for last frame")

                if plot_object["type"] == "scatter":
                    # get data
                    col_index_x = col_names.index(plot_object["obj"][0])
                    col_index_y = col_names.index(plot_object["obj"][1])
                    x = self.data[:, col_index_x]
                    y = self.data[:, col_index_y]

                    if self.initial_loop is True:
                        self.fc = np.repeat(
                            np.array(self.colors[0])[:, np.newaxis], self.n_pts, axis=1).T
                        for idx, subset in enumerate(self.subselections):
                            if subset.shape[0] != 0:
                                self.fc[subset] = self.colors[idx]
                        scat = self.axs[subplot_idx].scatter(x, y)
                        scat.set_facecolor(self.fc)
                    else:
                        self.axs[subplot_idx].collections[0].set_facecolors(
                            self.plot_dicts[subplot_idx]["fc"])

                    x_lims = self.axs[subplot_idx].get_xlim()
                    y_lims = self.axs[subplot_idx].get_ylim()

                    self.axs[subplot_idx].set_xlim(x_lims)
                    self.axs[subplot_idx].set_ylim(y_lims)
                    self.axs[subplot_idx].set_box_aspect(aspect=1)

                    plot_dict = {"type": "scatter",
                                 "subtype": "scatter",
                                 "subplot_idx": subplot_idx,
                                 "fc": self.fc,
                                 "ax": self.axs[subplot_idx]
                                 }
                    self.plot_dicts[subplot_idx] = plot_dict
                    # start Lasso selector
                    selector = LassoSelect(
                        plot_dicts=self.plot_dicts,
                        subplot_idx=subplot_idx,
                        colors=self.colors,
                        n_pts=self.n_pts,
                        alpha_other=self.alpha_other)
                    self.selectors.append(selector)
                    x_name = plot_object["obj"][0]
                    y_name = plot_object["obj"][1]
                    self.axs[subplot_idx].set_xlabel(x_name)
                    self.axs[subplot_idx].set_ylabel(y_name)
                    self.axs[subplot_idx].set_title(
                        f"Scatterplot of variables {x_name} and {y_name}")

                elif plot_object["type"] == "hist":
                    if plot_object["obj"] in col_names:
                        col_index = col_names.index(plot_object["obj"])
                        x = self.data[:, col_index]
                        # clear old histogram
                        self.axs[subplot_idx].clear()

                        # recolor preselected points
                        x_subselections = []
                        for subselection in self.plot_dicts[0]["subselections"]:
                            if subselection.shape[0] != 0:
                                x_subselections.append(x[subselection])
                            else:
                                x_subselections.append(np.array([]))
                        hist = self.axs[subplot_idx].hist(
                            x_subselections,
                            stacked=True,
                            picker=True,
                            color=self.colors[:len(x_subselections)])
                        y_lims = self.axs[subplot_idx].get_ylim()
                        self.axs[subplot_idx].set_ylim(y_lims)

                        self.axs[subplot_idx].set_box_aspect(aspect=1)
                        hist_variable_name = plot_object["obj"]
                        self.axs[subplot_idx].set_xlabel(hist_variable_name)
                        self.axs[subplot_idx].set_title(
                            f"Histogram of variable {hist_variable_name}")

                        if self.initial_loop is True:
                            self.fc = np.repeat(
                                np.array(self.colors[0])[:, np.newaxis], self.n_pts, axis=1).T

                            for idx, subset in enumerate(self.subselections):
                                if subset.shape[0] != 0:
                                    self.fc[subset] = self.colors[idx]

                            plot_dict = {"type": "hist",
                                         "subtype": "hist",
                                         "subplot_idx": subplot_idx,
                                         "ax": self.axs[subplot_idx],
                                         "data": self.data,
                                         "hist_feature": col_index,
                                         "feature_selection": self.feature_selection,
                                         "subselection_vars": self.subselection_vars,
                                         "subselections": self.subselections,
                                         "half_range": half_range,
                                         "fc": self.fc,
                                         "proj": proj}
                            self.plot_dicts[subplot_idx] = plot_dict
                            bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                     subplot_idx=subplot_idx,
                                                     feature_selection=self.feature_selection,
                                                     colors=self.colors,
                                                     half_range=self.half_range,
                                                     alpha_other=self.alpha_other)
                            self.plot_dicts[subplot_idx]["selector"] = bar_selector
                        else:
                            plot_dict = {"type": "hist",
                                         "subtype": "hist",
                                         "subplot_idx": subplot_idx,
                                         "ax": self.axs[subplot_idx],
                                         "data": self.data,
                                         "hist_feature": col_index,
                                         "feature_selection": self.feature_selection,
                                         "subselection_vars": self.subselection_vars,
                                         "subselections": self.subselections,
                                         "half_range": half_range,
                                         "fc": self.fc,
                                         "proj": proj,
                                         "selector": self.plot_dicts[subplot_idx]["selector"]}
                            self.plot_dicts[subplot_idx] = plot_dict
                            self.plot_dicts[subplot_idx]["selector"].disconnect(
                            )
                            bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                     subplot_idx=subplot_idx,
                                                     feature_selection=self.feature_selection,
                                                     colors=self.colors,
                                                     half_range=self.half_range,
                                                     alpha_other=self.alpha_other)
                            self.plot_dicts[subplot_idx]["selector"] = bar_selector
                    else:
                        print("Column not found")

            fig.canvas.draw()
            self.pause_var = tk.StringVar()
            fig.canvas.mpl_connect("key_press_event", accept)
            self.wait_variable(self.pause_var)

        self.destroy()

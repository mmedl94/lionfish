from checkbox_events import feature_checkbox_event, subselection_checkbox_event
from pytour_selectors import LassoSelect, DraggableAnnotation1d, DraggableAnnotation2d, BarSelect
from helpers import gram_schmidt
import tkinter as tk
from functools import partial
from datetime import datetime
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.style as mplstyle
import customtkinter as ctk


def interactive_tour(data, col_names, plot_objects, half_range=None, n_max_cols=None,
                     preselection=None, preselection_names=None, n_subsets=3):
    """Launch InteractiveTourInterface object"""
    app = InteractiveTourInterface(data,
                                   col_names,
                                   plot_objects,
                                   half_range,
                                   n_max_cols,
                                   preselection,
                                   preselection_names,
                                   n_subsets)
    app.mainloop()


class InteractiveTourInterface(ctk.CTk):
    def __init__(self, data, col_names, plot_objects, half_range, n_max_cols,
                 preselection, preselection_names, n_subsets):
        super().__init__()

        def accept(event):
            if event.key == "right" or event.key == "left":
                self.initial_loop = False

                for subplot_idx, _ in enumerate(self.plot_objects):
                    if "selector" in self.plot_dicts[subplot_idx]:
                        self.plot_dicts[subplot_idx]["selector"].disconnect()

                    if self.frame_vars[subplot_idx].get() != "":
                        if event.key == "right":
                            next_frame = int(
                                self.frame_vars[subplot_idx].get()) + 1
                            self.frame_vars[subplot_idx].set(str(next_frame))
                        if event.key == "left":
                            last_frame = int(
                                self.frame_vars[subplot_idx].get()) - 1
                            if last_frame < 0:
                                last_frame = 0
                            self.frame_vars[subplot_idx].set(str(last_frame))
                self.pause_var.set(1)

        self.title("Interactive tourr")
        self.data = data
        self.col_names = col_names
        self.half_range = half_range
        self.plot_objects = plot_objects
        self.displayed_tour = "Original tour"
        self.r = r

        if preselection is not False:
            if n_subsets < len(set(preselection)):
                self.n_subsets = len(set(preselection))
            else:
                self.n_subsets = int(n_subsets)
        else:
            self.n_subsets = int(n_subsets)
        self.preselection = np.array(preselection, dtype=int)-1
        self.preselection_names = preselection_names
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

        sidebar = ctk.CTkScrollableFrame(self)
        sidebar.grid(row=0, column=0, sticky="ns")

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
            checkbox.grid(row=feature_idx, column=0, pady=3, sticky="w")
        self.feature_selection = np.bool_(self.feature_selection)

        subselection_frame = ctk.CTkFrame(sidebar)
        subselection_frame.grid(row=1, column=0)

        self.subselection_vars = []
        self.subselections = []
        self.subset_names = []
        for subselection_idx in range(self.n_subsets):
            if preselection is not False:
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
                                       text="",
                                       width=24,
                                       command=partial(
                                           subselection_checkbox_event, self, subselection_idx),
                                       variable=check_var,
                                       onvalue=1,
                                       offvalue=0)
            checkbox.grid(row=subselection_idx, column=0,
                          pady=3, padx=0)

            if preselection_names is False or subselection_idx+1 > len(preselection_names):
                textvariable = tk.StringVar(
                    self, f"Subset {subselection_idx+1}")
            else:
                textvariable = tk.StringVar(
                    self, preselection_names[subselection_idx])
            self.subset_names.append(textvariable)
            textbox = ctk.CTkEntry(master=subselection_frame,
                                   textvariable=textvariable)
            textbox.grid(row=subselection_idx, column=1,
                         pady=3, padx=0, sticky="w")

        frame_selection_frame = ctk.CTkFrame(sidebar)
        frame_selection_frame.grid(row=2, column=0)

        self.frame_vars = []
        self.frame_textboxes = []
        for subplot_idx, plot_object in enumerate(plot_objects):
            self.plot_objects[subplot_idx]["og_obj"] = self.plot_objects[subplot_idx]["obj"]
            textvariable = tk.StringVar(self, "0")

            label = ctk.CTkLabel(master=frame_selection_frame,
                                 text=f"Plot #{subplot_idx+1}")
            label.grid(row=subplot_idx, column=0,
                       pady=3, padx=0, sticky="w")

            textbox = ctk.CTkEntry(master=frame_selection_frame,
                                   textvariable=textvariable,
                                   width=40)
            textbox.grid(row=subplot_idx, column=1,
                         pady=3, padx=0, sticky="w")

            self.frame_vars.append(textvariable)
            self.frame_textboxes.append(textbox)

        def update_frames_event(self):
            self.initial_loop = False
            self.pause_var.set(0)

        update_frames_button = ctk.CTkButton(master=frame_selection_frame,
                                             text="Update frames",
                                             command=partial(update_frames_event, self))
        update_frames_button.grid(
            row=subplot_idx+1, column=0, columnspan=2, pady=(3, 3), sticky="n")

        animation_frame = ctk.CTkFrame(sidebar)
        animation_frame.grid(row=3, column=0)

        self.animation_switch = tk.IntVar(self, 0)
        checkbox = ctk.CTkCheckBox(master=animation_frame,
                                   text="",
                                   width=24,
                                   variable=self.animation_switch,
                                   command=partial(update_frames_event, self),
                                   onvalue=1,
                                   offvalue=0)
        checkbox.grid(row=0, column=0, pady=3)

        self.fps_variable = tk.StringVar(self, "1")
        textbox = ctk.CTkEntry(master=animation_frame,
                               width=40,
                               textvariable=self.fps_variable)
        textbox.grid(row=0, column=1, pady=3)

        label = ctk.CTkLabel(master=animation_frame,
                             text="seconds")
        label.grid(row=0, column=2, pady=3)

        def save_event(self):
            save_dir = ctk.filedialog.askdirectory()
            now = datetime.now()
            now = now.strftime("%d_%m_%Y_%H_%M")
            if os.path.isdir(f"{save_dir}/{now}") is False:
                os.mkdir(f"{save_dir}/{now}")

            save_df = pd.DataFrame(
                self.plot_dicts[0]["subselections"]).T

            # Get subselection names
            save_df.columns = [subset_name.get()
                               for subset_name in self.subset_names]
            filename = f"{save_dir}/{now}/subselections.csv"
            save_df.to_csv(filename, index=False)
            for idx, plot_dict in enumerate(self.plot_dicts):
                if "proj" in plot_dict:
                    save_df = pd.DataFrame(
                        plot_dict["proj"][self.feature_selection])
                    save_df["original variables"] = np.array(
                        self.col_names)[self.feature_selection]
                    save_df = save_df.set_index("original variables")
                    filename = f"{save_dir}/{now}/projection_object_{idx+1}.csv"
                    save_df.to_csv(filename)

        save_button = ctk.CTkButton(master=sidebar,
                                    width=100,
                                    height=32,
                                    border_width=0,
                                    corner_radius=8,
                                    text="Save projections \n and subsets",
                                    command=partial(save_event, self))
        save_button.grid(row=4, column=0, pady=(3, 3), sticky="n")

        def run_local_tour(self):
            for idx, plot_object in enumerate(self.plot_objects):
                if plot_object["type"] == "1d_tour" or plot_object["type"] == "2d_tour":
                    new_proj = self.r.get_local_history(self.data[:2],
                                                        self.plot_dicts[idx]["proj"])
                    self.plot_objects[idx]["og_frame"] = int(
                        self.frame_vars[idx].get())
                    self.plot_objects[idx]["obj"] = new_proj
                    self.displayed_tour = "Local tour"
                    self.frame_vars[idx].set("0")
            self.initial_loop = False
            self.pause_var.set(0)

        local_tour_button = ctk.CTkButton(master=sidebar,
                                          text="Run local tour",
                                          command=partial(run_local_tour, self))
        local_tour_button.grid(row=5, column=0, pady=(3, 3), sticky="n")

        def reset_original_tour(self):
            for idx, plot_object in enumerate(self.plot_objects):
                if plot_object["type"] == "1d_tour" or plot_object["type"] == "2d_tour":
                    self.plot_objects[idx]["obj"] = plot_object["og_obj"]
                    self.frame_vars[idx].set(
                        str(self.plot_objects[idx]["og_frame"]))
                    self.displayed_tour = "Original tour"
            self.initial_loop = False
            self.pause_var.set(0)

        original_tour_button = ctk.CTkButton(master=sidebar,
                                             text="Reset original tour",
                                             command=partial(reset_original_tour, self))
        original_tour_button.grid(row=6, column=0, pady=(3, 3), sticky="n")

        # Get max number of frames
        self.n_frames = 0
        for plot_object in plot_objects:
            if isinstance(plot_object["obj"], np.ndarray):
                if plot_object["obj"].shape[-1] > self.n_frames:
                    self.n_frames = plot_object["obj"].shape[-1]
        self.frame = 0

        # resolve while loop in case of window closing
        def cleanup():
            self.frame = self.n_frames
            self.pause_var.set(1)
        self.protocol("WM_DELETE_WINDOW", cleanup)

        self.plot_dicts = [i for i, _ in enumerate(plot_objects)]
        self.initial_loop = True

        while self.frame < self.n_frames:
            for subplot_idx, plot_object in enumerate(plot_objects):
                frame = self.frame

                ####### 2d tour #######

                if plot_object["type"] == "2d_tour":
                    frame = int(self.frame_vars[subplot_idx].get())

                    if frame >= plot_object["obj"].shape[-1]-1:
                        frame = plot_object["obj"].shape[-1]-1
                        self.frame_vars[subplot_idx].set(str(frame))

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
                        n_pts=self.n_pts)
                    self.plot_dicts[subplot_idx]["selector"] = selector

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
                    self.axs[subplot_idx].set_title(
                        f"{self.displayed_tour}\n" +
                        f"Frame {frame} out of {n_frames}\n" +
                        f"Press right key for next frame\n" +
                        f"Press left key for last frame")

                ####### 1d tour #######

                if plot_object["type"] == "1d_tour":
                    frame = int(self.frame_vars[subplot_idx].get())
                    if frame >= plot_object["obj"].shape[-1]-1:
                        frame = plot_object["obj"].shape[-1]-1
                        self.frame_vars[subplot_idx].set(str(frame))

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
                                                 half_range=self.half_range)
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
                                     "fc": self.fc}
                        self.plot_dicts[subplot_idx] = plot_dict
                        bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                 subplot_idx=subplot_idx,
                                                 feature_selection=self.feature_selection,
                                                 colors=self.colors,
                                                 half_range=self.half_range)
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
                    self.axs[subplot_idx].set_title(
                        f"{self.displayed_tour}\n" +
                        f"Frame {frame} out of {n_frames}\n" +
                        f"Press right key for next frame\n" +
                        f"Press left key for last frame")

                ####### Scatterplot #######

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
                        self.frame_vars[subplot_idx].set("")
                        self.frame_textboxes[subplot_idx].configure(
                            state="disabled",
                            fg_color="grey")
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
                        n_pts=self.n_pts)
                    self.plot_dicts[subplot_idx]["selector"] = selector
                    x_name = plot_object["obj"][0]
                    y_name = plot_object["obj"][1]
                    self.axs[subplot_idx].set_xlabel(x_name)
                    self.axs[subplot_idx].set_ylabel(y_name)
                    self.axs[subplot_idx].set_title(
                        f"Scatterplot of variables {x_name} and {y_name}")

                ####### Histogram #######

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
                            self.frame_vars[subplot_idx].set("")
                            self.frame_textboxes[subplot_idx].configure(
                                state="disabled",
                                fg_color="grey")

                            for idx, subset in enumerate(self.subselections):
                                if subset.shape[0] != 0:
                                    self.fc[subset] = self.colors[idx]

                            plot_dict = {"type": "hist",
                                         "subtype": "hist",
                                         "subplot_idx": subplot_idx,
                                         "ax": self.axs[subplot_idx],
                                         "data": self.data,
                                         "hist_feature": col_index,
                                         "subselection_vars": self.subselection_vars,
                                         "subselections": self.subselections,
                                         "half_range": half_range,
                                         "fc": self.fc}
                            self.plot_dicts[subplot_idx] = plot_dict
                            bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                     subplot_idx=subplot_idx,
                                                     feature_selection=self.feature_selection,
                                                     colors=self.colors,
                                                     half_range=self.half_range)
                            self.plot_dicts[subplot_idx]["selector"] = bar_selector
                        else:
                            plot_dict = {"type": "hist",
                                         "subtype": "hist",
                                         "subplot_idx": subplot_idx,
                                         "ax": self.axs[subplot_idx],
                                         "data": self.data,
                                         "hist_feature": col_index,
                                         "subselection_vars": self.subselection_vars,
                                         "subselections": self.subselections,
                                         "half_range": half_range,
                                         "fc": self.fc,
                                         "selector": self.plot_dicts[subplot_idx]["selector"]}
                            self.plot_dicts[subplot_idx] = plot_dict
                            self.plot_dicts[subplot_idx]["selector"].disconnect(
                            )
                            bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                                     subplot_idx=subplot_idx,
                                                     feature_selection=self.feature_selection,
                                                     colors=self.colors,
                                                     half_range=self.half_range)
                            self.plot_dicts[subplot_idx]["selector"] = bar_selector
                    else:
                        print("Column not found")

                ####### categorical cluster interface #######

                elif plot_object["type"] == "cat_clust_interface":
                    self.axs[subplot_idx].set_box_aspect(aspect=1)

                    # Get data
                    # Initialize data array
                    cat_clust_data = np.empty(
                        (len(self.feature_selection), int(n_subsets)))

                    # get ratios
                    all_pos = np.sum(self.data, axis=0)
                    for subset_idx, subset in enumerate(self.subselections):
                        if subset.shape[0] != 0:
                            all_pos_subset = np.sum(self.data[subset], axis=0)
                            cat_clust_data[:,
                                           subset_idx] = all_pos_subset/all_pos
                        else:
                            cat_clust_data[:, subset_idx] = np.zeros(
                                len(self.feature_selection))

                    var_ids = np.repeat(np.arange(sum(self.feature_selection)),
                                        int(n_subsets))
                    cat_clust_data = cat_clust_data.flatten()

                    # make cluster color scheme
                    clust_colors = np.tile(self.colors,
                                           (len(self.feature_selection), 1))
                    clust_colors = np.concatenate((clust_colors,
                                                  np.ones((clust_colors.shape[0], 1))),
                                                  axis=1)

                    clust_ids = np.arange(int(n_subsets))
                    clust_ids = np.tile(clust_ids, len(self.feature_selection))

                    # current cluster selection
                    for subselection_id, subselection_var in enumerate(self.subselection_vars):
                        if subselection_var.get() == 1:
                            selected_cluster = subselection_id

                    selected = np.where(
                        clust_ids == selected_cluster)[0]
                    not_selected = np.where(
                        clust_ids != selected_cluster)[0]

                    clust_colors[not_selected, -1] = 0.2
                    clust_colors[selected, -1] = 1

                    feature_selection_bool = np.repeat(
                        self.feature_selection, n_subsets)

                    if self.initial_loop is False:
                        self.axs[subplot_idx].clear()
                        self.frame_vars[subplot_idx].set("")
                        self.frame_textboxes[subplot_idx].configure(
                            state="disabled",
                            fg_color="grey")

                    scat = self.axs[subplot_idx].scatter(
                        cat_clust_data[feature_selection_bool],
                        var_ids,
                        c=clust_colors[feature_selection_bool])

                    y_tick_labels = np.array(col_names)[self.feature_selection]
                    self.axs[subplot_idx].set_yticks(
                        np.arange(0, sum(self.feature_selection)))
                    self.axs[subplot_idx].set_yticklabels(y_tick_labels)

                    plot_dict = {"type": "cat_clust_interface",
                                 "subtype": "cat_clust_interface",
                                 "subplot_idx": subplot_idx,
                                 "ax": self.axs[subplot_idx],
                                 "data": self.data,
                                 "feature_selection": self.feature_selection,
                                 "cat_clust_data": cat_clust_data,
                                 "col_names": col_names,
                                 "subselection_vars": self.subselection_vars,
                                 "subselections": self.subselections,
                                 "half_range": half_range}
                    self.plot_dicts[subplot_idx] = plot_dict

            fig.canvas.draw()

            def wait(self):
                var = tk.IntVar()
                self.after(
                    int(float(self.fps_variable.get()) * 1000), var.set, 1)
                self.wait_variable(var)

            if self.animation_switch.get() == 1:
                wait(self)
                self.initial_loop = False
                for subplot_idx, _ in enumerate(self.plot_objects):
                    self.plot_dicts[subplot_idx]["selector"].disconnect()
                    if self.frame_vars[subplot_idx].get() != "":
                        next_frame = int(
                            self.frame_vars[subplot_idx].get()) + 1
                        self.frame_vars[subplot_idx].set(str(next_frame))

            else:
                self.pause_var = tk.StringVar()
                fig.canvas.mpl_connect("key_press_event", accept)
                self.wait_variable(self.pause_var)

        self.destroy()

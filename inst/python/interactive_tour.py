from checkbox_events import feature_checkbox_event, subselection_checkbox_event
from pytour_selectors import LassoSelect, DraggableAnnotation1d, DraggableAnnotation2d, BarSelect
from helpers import gram_schmidt
import tkinter as tk
from functools import partial
from itertools import product
from datetime import datetime
import os
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.style as mplstyle
from statsmodels.graphics.mosaicplot import mosaic
import customtkinter as ctk
import seaborn as sns


def interactive_tour(data, col_names, plot_objects, half_range=None, n_max_cols=None,
                     preselection=None, preselection_names=None, n_subsets=3, size=10):
    """Launch InteractiveTourInterface object"""
    app = InteractiveTourInterface(data,
                                   col_names,
                                   plot_objects,
                                   half_range,
                                   n_max_cols,
                                   preselection,
                                   preselection_names,
                                   n_subsets,
                                   size)
    app.mainloop()


class InteractiveTourInterface(ctk.CTk):
    def __init__(self, data, col_names, plot_objects, half_range, n_max_cols,
                 preselection, preselection_names, n_subsets, size):
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
        self.reset_selection_check = False
        self.size = size

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
        self.colors = [[r, g, b, 1.0] for [r, g, b] in self.colors]

        if not isinstance(plot_objects, list):
            plot_objects = [plot_objects]

        if half_range is None:
            print("Using adaptive half_range")
        else:
            print(f"Using half_range of {half_range}")

        self.limits = 1
        self.n_pts = self.data.shape[0]
        # Initialize self.obs_idx with all obs
        self.obs_idx_ = np.arange(0, self.data.shape[0])
        if len(plot_objects) == 1:
            fig, self.axs = plt.subplots(1, 1, figsize=(self.size, self.size))
            self.axs = [self.axs]
        else:
            n_plots = len(plot_objects)
            if n_max_cols is None:
                n_max_cols = 3
            n_cols = int(min(n_max_cols, n_plots))
            n_rows = int((n_plots + n_cols - 1) // n_cols)

            fig, self.axs = plt.subplots(
                n_rows, n_cols, figsize=(self.size*n_cols,
                                         self.size*n_rows),
                layout="compressed")
            self.axs = self.axs.flatten()
            for i in range(n_plots, len(self.axs)):
                fig.delaxes(self.axs[i])
            self.axs = self.axs[:n_plots]

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.get_tk_widget().grid(row=0, column=1, sticky="n")

        sidebar = ctk.CTkScrollableFrame(self)
        sidebar.grid(row=0, column=0, sticky="ns")

        ###### Feature selections ######

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

        ###### Subselections ######

        def update_colors(self, subselection_idx):
            col_array = np.array(self.colors[subselection_idx])
            selected_row_idx = np.where(np.all(self.fc[:, :3] == col_array[:3],
                                               axis=1))

            if self.colors[subselection_idx][-1] == 1:
                self.colors[subselection_idx][-1] = 0.3
                self.fc[selected_row_idx, -1] = 0.1
            else:
                self.colors[subselection_idx][-1] = 1
                self.fc[selected_row_idx, -1] = 1
            for subplot_idx, plot_dict in enumerate(self.plot_dicts):
                if plot_dict["subtype"] in ["1d_tour", "2d_tour"]:
                    self.plot_dicts[subplot_idx]["update_plot"] = False

            self.pause_var.set(0)

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

            self.orig_subselections = self.subselections.copy()
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

            color_box = ctk.CTkButton(master=subselection_frame,
                                      text="",
                                      width=24,
                                      height=24,
                                      hover=False,
                                      fg_color=matplotlib.colors.rgb2hex(
                                          self.colors[subselection_idx]),
                                      command=partial(update_colors,
                                                      self,
                                                      subselection_idx))
            color_box.grid(row=subselection_idx, column=2,
                           pady=3, padx=2, sticky="w")

        def reset_selection(self):
            self.subselections = self.orig_subselections.copy()
            self.fc = self.original_fc.copy()
            for subplot_idx, _ in enumerate(self.plot_objects):
                if "selector" in self.plot_dicts[subplot_idx]:
                    self.plot_dicts[subplot_idx]["selector"].disconnect()

            self.reset_selection_check = True
            self.pause_var.set(0)

        reset_selection_button = ctk.CTkButton(master=subselection_frame,
                                               text="Reset original selection",
                                               command=partial(reset_selection, self))
        reset_selection_button.grid(
            row=subselection_idx+1, column=0, columnspan=2, pady=3, sticky="n")

        ###### Frame selectors ######

        frame_selection_frame = ctk.CTkFrame(sidebar)
        frame_selection_frame.grid(row=2, column=0)

        self.frame_vars = []
        self.frame_textboxes = []
        for subplot_idx, plot_object in enumerate(plot_objects):
            self.plot_objects[subplot_idx]["og_obj"] = self.plot_objects[subplot_idx]["obj"]
            textvariable = tk.StringVar(self, "")

            label = ctk.CTkLabel(master=frame_selection_frame,
                                 text=f"Plot #{subplot_idx+1}")
            label.grid(row=subplot_idx, column=0,
                       pady=3, padx=0, sticky="w")

            textbox = ctk.CTkEntry(master=frame_selection_frame,
                                   textvariable=textvariable,
                                   width=40,
                                   state="disabled",
                                   fg_color="grey")
            textbox.grid(row=subplot_idx, column=1,
                         pady=3, padx=0, sticky="w")

            self.frame_vars.append(textvariable)
            self.frame_textboxes.append(textbox)

        def update_frames_event(self):
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

        ###### Save button ######

        def save_event(self):
            save_dir = ctk.filedialog.askdirectory()
            now = datetime.now()
            now = now.strftime("%d_%m_%Y_%H_%M")
            if os.path.isdir(f"{save_dir}/{now}") is False:
                os.mkdir(f"{save_dir}/{now}")

            save_df = pd.DataFrame(
                self.subselections.T)

            # Get subselection names
            save_df.columns = [subset_name.get()
                               for subset_name in self.subset_names]
            save_df.to_csv(f"{save_dir}/{now}/subselections.csv", index=False)
            fig.savefig(f"{save_dir}/{now}/figure.png")
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

        ##### New tours #####

        tour_types = ["Local tour",
                      "Guided tour - holes",
                      "Guided tour - holes - better",
                      "Guided tour - lda"]

        self.selected_tour_type = ctk.StringVar(
            value="Local tour")
        tour_menu = ctk.CTkComboBox(master=sidebar,
                                    values=tour_types,
                                    variable=self.selected_tour_type)
        tour_menu.grid(row=5, column=0, pady=(3, 3), sticky="n")

        def run_tour(self):
            for idx, plot_object in enumerate(self.plot_objects):
                if plot_object["type"] == "1d_tour" or plot_object["type"] == "2d_tour":

                    if plot_object["type"] == "1d_tour":
                        dimension = 1
                    elif plot_object["type"] == "2d_tour":
                        dimension = 2
                    full_array = np.zeros(
                        (self.feature_selection.shape[0], dimension))

                    if self.selected_tour_type.get() == "Local tour":
                        new_proj = self.r.get_local_history(
                            self.data[:2, self.feature_selection],
                            self.plot_dicts[idx]["proj"][self.feature_selection])
                        self.displayed_tour = self.selected_tour_type.get()

                    elif self.selected_tour_type.get() == "Guided tour - holes":
                        new_proj = self.r.get_guided_holes_history(
                            self.data[:, self.feature_selection],
                            dimension)
                        self.displayed_tour = self.selected_tour_type.get()

                    elif self.selected_tour_type.get() == "Guided tour - holes - better":
                        new_proj = self.r.get_guided_holes_better_history(
                            self.data[:, self.feature_selection],
                            dimension)
                        self.displayed_tour = self.selected_tour_type.get()

                    elif self.selected_tour_type.get() == "Guided tour - lda":
                        subselection_idxs = np.zeros(self.n_pts, dtype=int)
                        for subselection_idx, arr in enumerate(self.subselections):
                            if arr.shape[0] != 0:
                                subselection_idxs[arr] = subselection_idx + 1

                        new_proj = self.r.get_guided_lda_history(self.data[:, self.feature_selection],
                                                                 subselection_idxs,
                                                                 dimension)
                        self.displayed_tour = self.selected_tour_type.get()

                    full_array = np.tile(
                        full_array[:, :, np.newaxis], (1, 1, new_proj.shape[2]))
                    full_array[self.feature_selection] = new_proj

                    self.plot_objects[idx]["og_frame"] = int(
                        self.frame_vars[idx].get())
                    self.plot_objects[idx]["obj"] = full_array
                    self.frame_vars[idx].set("0")
            self.pause_var.set(0)

        run_tour_button = ctk.CTkButton(master=sidebar,
                                        text="Run tour",
                                        command=partial(run_tour, self))
        run_tour_button.grid(row=6, column=0, pady=(3, 3), sticky="n")

        ###### Reset tour button ######

        def reset_original_tour(self):
            for idx, plot_object in enumerate(self.plot_objects):
                if plot_object["type"] == "1d_tour" or plot_object["type"] == "2d_tour":
                    self.plot_objects[idx]["obj"] = plot_object["og_obj"]
                    self.frame_vars[idx].set(
                        str(self.plot_objects[idx]["og_frame"]))
                    self.displayed_tour = "Original tour"
            self.pause_var.set(0)

        original_tour_button = ctk.CTkButton(master=sidebar,
                                             text="Reset original tour",
                                             command=partial(reset_original_tour, self))
        original_tour_button.grid(row=7, column=0, pady=(3, 3), sticky="n")

        row_tracker = 7

        ###### Metric menu ######
        metrics = ["Intra cluster fraction",
                   "Intra feature fraction",
                   "Total fraction"]

        plot_types_w_metric = ["heatmap",
                               "cat_clust_interface"]

        def update_metric_event(self, selection):
            self.pause_var.set(0)

        # check if a single interface used a metric
        need_metric = False
        for subplot_idx, plot_object in enumerate(plot_objects):
            if plot_object["type"] in plot_types_w_metric:
                need_metric = True

        if need_metric:
            metric_selection_frame = ctk.CTkFrame(sidebar)
            metric_selection_frame.grid(row=row_tracker, column=0)
            row_tracker += 1

            menu_tracker = 0
            self.metric_vars = []
            for subplot_idx, plot_object in enumerate(plot_objects):
                metric_var = tk.StringVar("")
                if str(plot_object["obj"]) in set(metrics):
                    metric_var.set(plot_object["obj"])
                else:
                    metric_var.set("Intra cluster fraction of positive")

                label = ctk.CTkLabel(master=metric_selection_frame,
                                     text=f"Plot #{subplot_idx+1}")
                label.grid(
                    row=menu_tracker, column=0, pady=(3, 3), sticky="n")

                metric_selection_menu = ctk.CTkComboBox(master=metric_selection_frame,
                                                        values=metrics,
                                                        command=partial(
                                                            update_metric_event, self),
                                                        variable=metric_var)
                metric_selection_menu.grid(
                    row=menu_tracker, column=1, pady=(3, 3), sticky="n")

                if str(plot_object["type"]) not in set(plot_types_w_metric):
                    metric_selection_menu.configure(state="disabled",
                                                    fg_color="grey")

                menu_tracker += 1

                self.metric_vars.append(metric_var)

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

        self.plot_dicts = [{} for i, _ in enumerate(plot_objects)]
        self.initial_loop = True

        ###### Plot loop ######
        self.blit = 0
        self.last_frame = -1
        while self.frame < self.n_frames:
            self.pause_var = tk.StringVar(value=42)
            for subplot_idx, plot_object in enumerate(plot_objects):
                frame = self.frame

                ####### 2d tour #######

                if plot_object["type"] == "2d_tour":
                    if self.initial_loop is True:
                        frame = 0
                    else:
                        frame = int(self.frame_vars[subplot_idx].get())

                    if frame >= plot_object["obj"].shape[-1]-1:
                        frame = plot_object["obj"].shape[-1]-1
                        self.frame_vars[subplot_idx].set(str(frame))

                    if "update_plot" in self.plot_dicts[subplot_idx]:
                        update_plot = self.plot_dicts[subplot_idx]["update_plot"]
                        self.plot_dicts[subplot_idx]["update_plot"] = True
                    else:
                        update_plot = True

                    if update_plot is True:
                        if self.reset_selection_check is False:
                            proj = np.copy(
                                plot_object["obj"][:, :, frame])
                        else:
                            proj = self.plot_dicts[subplot_idx]["proj"]
                            self.reset_selection_check = False
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
                            self.frame_vars[subplot_idx].set("0")
                            self.frame_textboxes[subplot_idx].configure(
                                state="normal",
                                fg_color="white")
                            self.fc = np.repeat(
                                np.array(self.colors[0])[:, np.newaxis], self.n_pts, axis=1).T
                            for idx, subset in enumerate(self.subselections):
                                if subset.shape[0] != 0:
                                    self.fc[subset] = self.colors[idx]
                            scat = self.axs[subplot_idx].scatter(
                                x, y, animated=True)
                            scat.set_facecolor(self.fc)
                            self.original_fc = self.fc.copy()
                            self.axs[subplot_idx].plot(circle_prj.iloc[:, 0],
                                                       circle_prj.iloc[:, 1], color="grey")
                        else:
                            # clear old arrows and text
                            for patch_idx, _ in enumerate(self.axs[subplot_idx].patches):
                                self.axs[subplot_idx].patches[0].remove()
                                self.axs[subplot_idx].texts[0].remove()
                            self.plot_dicts[subplot_idx]["draggable_annot"].disconnect(
                            )
                            self.plot_dicts[subplot_idx]["selector"].disconnect(
                            )
                            # update scatterplot
                            self.plot_dicts[subplot_idx]["scat"].set_offsets(
                                np.array([x, y]).T)

                            scat = self.plot_dicts[subplot_idx]["ax"].collections[0]
                            scat.set_facecolors(self.fc)

                        self.axs[subplot_idx].set_xlim(-self.limits *
                                                       1.1, self.limits*1.1)
                        self.axs[subplot_idx].set_ylim(-self.limits *
                                                       1.1, self.limits*1.1)
                        self.axs[subplot_idx].set_xticks([])
                        self.axs[subplot_idx].set_yticks([])
                        self.axs[subplot_idx].set_aspect("equal")

                        plot_dict = {"type": "scatter",
                                     "subtype": "2d_tour",
                                     "subplot_idx": subplot_idx,
                                     "ax": self.axs[subplot_idx],
                                     "data": self.data,
                                     "scat": scat,
                                     "feature_selection": self.feature_selection,
                                     "proj": proj,
                                     "update_plot": True
                                     }
                        self.plot_dicts[subplot_idx] = plot_dict

                        # start Lasso selector
                        selector = LassoSelect(
                            parent=self,
                            plot_dicts=self.plot_dicts,
                            subplot_idx=subplot_idx,
                            colors=self.colors,
                            n_pts=self.n_pts,
                            pause_var=self.pause_var)
                        self.plot_dicts[subplot_idx]["selector"] = selector

                        plot_dict["draggable_annot"] = DraggableAnnotation2d(
                            self,
                            self.data,
                            self.plot_dicts[subplot_idx]["proj"],
                            self.axs[subplot_idx],
                            self.plot_dicts[subplot_idx]["scat"],
                            half_range,
                            self.feature_selection,
                            col_names,
                            self.plot_dicts[subplot_idx],
                            self.plot_dicts,
                            subplot_idx,
                            self.pause_var)

                        n_frames = plot_object["obj"].shape[-1]-1
                        self.axs[subplot_idx].set_title(
                            f"{self.displayed_tour}\n" +
                            f"Frame {frame} out of {n_frames}\n" +
                            "Press right key for next frame\n" +
                            "Press left key for last frame")
                    else:
                        self.plot_dicts[subplot_idx]["scat"].set_facecolors(
                            self.fc)
                        self.plot_dicts[subplot_idx]["selector"].pause_var = self.pause_var

                ####### 1d tour #######

                if plot_object["type"] == "1d_tour":
                    if "update_plot" in self.plot_dicts[subplot_idx]:
                        update_plot = self.plot_dicts[subplot_idx]["update_plot"]
                        self.plot_dicts[subplot_idx]["update_plot"] = True
                    else:
                        update_plot = True
                    if self.initial_loop is True:
                        frame = 0
                    else:
                        frame = int(self.frame_vars[subplot_idx].get())

                    if frame >= plot_object["obj"].shape[-1]-1:
                        frame = plot_object["obj"].shape[-1]-1
                        self.frame_vars[subplot_idx].set(str(frame))

                    if update_plot is True:
                        proj = np.copy(
                            plot_object["obj"][:, :, frame])

                        data_subset = self.data[:, self.feature_selection]
                        proj_subet = proj[self.feature_selection][:, 0]
                        proj_subet = proj_subet / \
                            np.linalg.norm(proj_subet)
                        x = np.matmul(data_subset, proj_subet)
                        x = x/half_range
                    else:
                        proj = self.plot_dicts[subplot_idx]["proj"]
                        x = self.plot_dicts[subplot_idx]["x"]

                    self.axs[subplot_idx].clear()

                    # check if there are preselected points and update plot
                    # recolor preselected points
                    x_subselections = []
                    for subselection in self.subselections:
                        if subselection.shape[0] != 0:
                            x_subselections.append(x[subselection])
                        else:
                            x_subselections.append(np.array([]))
                    hist = self.axs[subplot_idx].hist(
                        x_subselections,
                        stacked=True,
                        picker=True,
                        color=self.colors[:len(x_subselections)],
                        animated=True)
                    y_lims = self.axs[subplot_idx].get_ylim()
                    self.axs[subplot_idx].set_ylim(y_lims)

                    if self.initial_loop is True:
                        self.frame_vars[subplot_idx].set("0")
                        self.frame_textboxes[subplot_idx].configure(
                            state="normal",
                            fg_color="white")
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
                                     "half_range": half_range,
                                     "proj": proj}
                        self.plot_dicts[subplot_idx] = plot_dict
                        bar_selector = BarSelect(parent=self,
                                                 plot_dicts=self.plot_dicts,
                                                 subplot_idx=subplot_idx,
                                                 feature_selection=self.feature_selection,
                                                 colors=self.colors,
                                                 half_range=self.half_range,
                                                 pause_var=self.pause_var)
                    else:
                        self.plot_dicts[subplot_idx]["draggable_annot"].disconnect(
                        )
                        self.plot_dicts[subplot_idx]["draggable_annot"].remove(
                        )
                        plot_dict = {"type": "hist",
                                     "subtype": "1d_tour",
                                     "subplot_idx": subplot_idx,
                                     "ax": self.axs[subplot_idx],
                                     "data": self.data,
                                     "feature_selection": self.feature_selection,
                                     "half_range": half_range,
                                     "proj": proj}
                        self.plot_dicts[subplot_idx] = plot_dict
                        bar_selector = BarSelect(parent=self,
                                                 plot_dicts=self.plot_dicts,
                                                 subplot_idx=subplot_idx,
                                                 feature_selection=self.feature_selection,
                                                 colors=self.colors,
                                                 half_range=self.half_range,
                                                 pause_var=self.pause_var)
                    self.plot_dicts[subplot_idx]["selector"] = bar_selector

                    draggable_arrows_1d = DraggableAnnotation1d(
                        self,
                        self.data,
                        self.plot_dicts,
                        subplot_idx,
                        hist,
                        half_range,
                        self.feature_selection,
                        self.colors,
                        col_names,
                        self.pause_var)

                    self.plot_dicts[subplot_idx]["draggable_annot"] = draggable_arrows_1d

                    n_frames = plot_object["obj"].shape[-1]-1
                    self.axs[subplot_idx].set_xticks([])
                    self.axs[subplot_idx].set_yticks([])
                    self.axs[subplot_idx].set_xlim(-1, 1)
                    self.axs[subplot_idx].set_title(
                        f"{self.displayed_tour}\n" +
                        f"Frame {frame} out of {n_frames}\n" +
                        "Press right key for next frame\n" +
                        "Press left key for last frame")

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
                        scat = self.axs[subplot_idx].scatter(x, y,
                                                             animated=True)
                        scat.set_facecolor(self.fc)
                    else:
                        self.axs[subplot_idx].collections[0].set_facecolors(
                            self.fc)
                        self.plot_dicts[subplot_idx]["selector"].disconnect()

                    x_lims = self.axs[subplot_idx].get_xlim()
                    y_lims = self.axs[subplot_idx].get_ylim()

                    self.axs[subplot_idx].set_xlim(x_lims)
                    self.axs[subplot_idx].set_ylim(y_lims)
                    plot_dict = {"type": "scatter",
                                 "subtype": "scatter",
                                 "subplot_idx": subplot_idx,
                                 "data": self.data,
                                 "ax": self.axs[subplot_idx]
                                 }
                    self.plot_dicts[subplot_idx] = plot_dict
                    # start Lasso selector
                    selector = LassoSelect(
                        parent=self,
                        plot_dicts=self.plot_dicts,
                        subplot_idx=subplot_idx,
                        colors=self.colors,
                        n_pts=self.n_pts,
                        pause_var=self.pause_var)
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
                        for subselection in self.subselections:
                            if subselection.shape[0] != 0:
                                x_subselections.append(x[subselection])
                            else:
                                x_subselections.append(np.array([]))
                        hist = self.axs[subplot_idx].hist(
                            x_subselections,
                            stacked=True,
                            picker=True,
                            color=self.colors[:len(x_subselections)],
                            animated=True)
                        y_lims = self.axs[subplot_idx].get_ylim()
                        self.axs[subplot_idx].set_ylim(y_lims)

                        hist_variable_name = plot_object["obj"]
                        self.axs[subplot_idx].set_xlabel(hist_variable_name)
                        self.axs[subplot_idx].set_title(
                            f"Histogram of variable {hist_variable_name}")
                        self.axs[subplot_idx].set_xticks([])

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
                                         "half_range": half_range}
                            self.plot_dicts[subplot_idx] = plot_dict
                            bar_selector = BarSelect(parent=self,
                                                     plot_dicts=self.plot_dicts,
                                                     subplot_idx=subplot_idx,
                                                     feature_selection=self.feature_selection,
                                                     colors=self.colors,
                                                     half_range=self.half_range,
                                                     pause_var=self.pause_var)
                            self.plot_dicts[subplot_idx]["selector"] = bar_selector
                        else:
                            plot_dict = {"type": "hist",
                                         "subtype": "hist",
                                         "subplot_idx": subplot_idx,
                                         "ax": self.axs[subplot_idx],
                                         "data": self.data,
                                         "hist_feature": col_index,
                                         "feature_selection": self.feature_selection,
                                         "half_range": half_range,
                                         "selector": self.plot_dicts[subplot_idx]["selector"]}
                            self.plot_dicts[subplot_idx] = plot_dict
                            self.plot_dicts[subplot_idx]["selector"].disconnect(
                            )
                            bar_selector = BarSelect(parent=self,
                                                     plot_dicts=self.plot_dicts,
                                                     subplot_idx=subplot_idx,
                                                     feature_selection=self.feature_selection,
                                                     colors=self.colors,
                                                     half_range=self.half_range,
                                                     pause_var=self.pause_var)
                            self.plot_dicts[subplot_idx]["selector"] = bar_selector
                    else:
                        print("Column not found")

                ####### categorical cluster interface #######

                elif plot_object["type"] == "cat_clust_interface":
                    # Get data
                    # Initialize data array
                    cat_clust_data = np.empty(
                        (len(self.feature_selection), int(n_subsets)))
                    cur_metric_var = self.metric_vars[subplot_idx].get()
                    # get ratios
                    all_pos = np.sum(self.data, axis=0)
                    for subset_idx, subset in enumerate(self.subselections):
                        if subset.shape[0] != 0:
                            all_pos_subset = np.sum(self.data[subset], axis=0)
                            if cur_metric_var == "Intra cluster fraction":
                                cat_clust_data[:,
                                               subset_idx] = all_pos_subset/self.data[subset].shape[0]
                            elif cur_metric_var == "Intra feature fraction":
                                cat_clust_data[:,
                                               subset_idx] = all_pos_subset/all_pos
                            elif cur_metric_var == "Total fraction":
                                cat_clust_data[:,
                                               subset_idx] = all_pos_subset/self.data.shape[0]
                        else:
                            cat_clust_data[:, subset_idx] = np.zeros(
                                len(self.feature_selection))

                    var_ids = np.repeat(np.arange(sum(self.feature_selection)),
                                        int(n_subsets))
                    cat_clust_data = cat_clust_data.flatten()

                    clust_ids = np.arange(n_subsets)
                    clust_ids = np.tile(clust_ids, len(self.feature_selection))

                    # current cluster selection
                    for subselection_id, subselection_var in enumerate(self.subselection_vars):
                        if subselection_var.get() == 1:
                            selected_cluster = subselection_id

                    feature_selection_bool = np.repeat(
                        self.feature_selection, n_subsets)

                    if self.initial_loop is False:
                        self.axs[subplot_idx].clear()

                    x = cat_clust_data[feature_selection_bool]

                    # Sort to display inter cluster max at the top
                    sort_idx = np.arange(
                        selected_cluster, x.shape[0], n_subsets, dtype=int)
                    ranked_vars = np.argsort(x[sort_idx])[::-1]
                    sorting_helper = np.arange(x.shape[0])
                    sorting_helper = sorting_helper.reshape(
                        sort_idx.shape[0], int(n_subsets))
                    sorting_helper = sorting_helper[ranked_vars].flatten()

                    # flip var_ids so most important is on top
                    var_ids = np.flip(var_ids)

                    # Get coloration scheme
                    fc = np.tile(self.colors,
                                 (len(self.feature_selection), 1))

                    scat = self.axs[subplot_idx].scatter(
                        x[sorting_helper],
                        var_ids,
                        c=fc[sorting_helper]
                    )

                    y_tick_labels = np.array(col_names)[self.feature_selection]
                    y_tick_labels = y_tick_labels[ranked_vars]
                    # flip so that labels agree with var_ids
                    y_tick_labels = np.flip(y_tick_labels)

                    self.axs[subplot_idx].set_yticks(
                        np.arange(0, sum(self.feature_selection)))
                    self.axs[subplot_idx].set_yticklabels(y_tick_labels)
                    self.axs[subplot_idx].set_xlabel(
                        cur_metric_var)

                    if self.subselections[selected_cluster].shape[0] == 0:
                        fraction_of_total = 0
                    else:
                        subset_size = self.data[self.subselections[selected_cluster]].shape[0]
                        fraction_of_total = (
                            subset_size/self.data.shape[0])*100
                    title = f"{subset_size} obersvations - ({fraction_of_total:.2f}%)"
                    self.axs[subplot_idx].set_title(title)

                    plot_dict = {"type": "cat_clust_interface",
                                 "subtype": cur_metric_var,
                                 "subplot_idx": subplot_idx,
                                 "ax": self.axs[subplot_idx],
                                 "data": self.data,
                                 "feature_selection": self.feature_selection,
                                 "cat_clust_data": cat_clust_data,
                                 "col_names": col_names,
                                 "half_range": half_range}
                    self.plot_dicts[subplot_idx] = plot_dict

                ####### Mosaic plot #######

                elif plot_object["type"] == "mosaic":
                    mosaic_data = np.empty(
                        (len(self.feature_selection), int(n_subsets)))
                    non_empty_sets = []
                    for subset_idx, subset in enumerate(self.subselections):
                        if subset.shape[0] != 0:
                            mosaic_data[:,
                                        subset_idx] = self.data[subset].sum(axis=0)
                            non_empty_sets.append(True)
                        else:
                            mosaic_data[:, subset_idx] = np.zeros(
                                len(self.feature_selection))
                            non_empty_sets.append(False)

                    mosaic_data = mosaic_data[self.feature_selection]
                    mosaic_data = mosaic_data[:, non_empty_sets]

                    y_tick_labels = np.array(col_names)[self.feature_selection]
                    x_tick_labels = np.array([subselection_var.get()
                                              for subselection_var in self.subset_names])
                    x_tick_labels = x_tick_labels[non_empty_sets]
                    if plot_object["obj"] == "subgroups_on_y":
                        tuples = list(
                            product(y_tick_labels, x_tick_labels))
                    else:
                        tuples = list(
                            product(x_tick_labels, y_tick_labels))

                    index = pd.MultiIndex.from_tuples(
                        tuples, names=["first", "second"])
                    mosaic_data = pd.Series(
                        mosaic_data.flatten(), index=index)

                    if self.initial_loop is False:
                        self.axs[subplot_idx].clear()
                        self.axs[subplot_idx].set_in_layout(True)

                    mosaic_colors = np.array(self.colors)[non_empty_sets]
                    color_dict = {}
                    if plot_object["obj"] == "subgroups_on_y":
                        unique_levels = mosaic_data.index.get_level_values(
                            "second").unique()
                        color_mapping = dict(zip(unique_levels, mosaic_colors))
                        for idx in mosaic_data.index:
                            # Get the color for the current second level value
                            color = color_mapping[idx[1]]
                            # Map the index tuple to its corresponding color
                            color_dict[idx] = {"color": color}
                    else:
                        unique_levels = mosaic_data.index.get_level_values(
                            "first").unique()
                        color_mapping = dict(zip(unique_levels, mosaic_colors))
                        for idx in mosaic_data.index:
                            # Get the color for the current 'second' level value
                            color = color_mapping[idx[0]]
                            # Map the index tuple to its corresponding color
                            color_dict[idx] = {"color": color}
                    mosaic(mosaic_data,
                           ax=self.axs[subplot_idx],
                           properties=color_dict,
                           gap=0.01)

                    xlabels = self.axs[subplot_idx].get_xticklabels()
                    self.axs[subplot_idx].set_xticklabels(xlabels,
                                                          rotation=90)

                    # remove extra plots
                    twinaxs = self.axs[subplot_idx].twinx()
                    remove_pos = twinaxs.get_position().bounds
                    twinaxs.remove()
                    for axs_idx, axs in enumerate(fig.get_axes()):
                        if axs.get_position().bounds == remove_pos:
                            if axs != self.axs[subplot_idx]:
                                axs.remove()

                    for patch in self.axs[subplot_idx].patches:
                        patch.set_animated(True)
                    for text in self.axs[subplot_idx].texts:
                        text.remove()

                    plot_dict = {"type": "mosaic",
                                 "subtype": "mosaic",
                                 "subplot_idx": subplot_idx,
                                 "ax": self.axs[subplot_idx],
                                 "data": self.data,
                                 "feature_selection": self.feature_selection,
                                 "mosaic_data": mosaic_data,
                                 "col_names": col_names,
                                 "half_range": half_range,
                                 "subset_names": self.subset_names}
                    self.plot_dicts[subplot_idx] = plot_dict

                ####### heatmap plot #######

                elif plot_object["type"] == "heatmap":
                    heatmap_data = np.empty(
                        (len(self.feature_selection), int(n_subsets)))
                    cur_metric_var = self.metric_vars[subplot_idx].get()
                    # get ratios
                    all_pos = np.sum(self.data, axis=0)
                    non_empty_sets = []

                    for subset_idx, subset in enumerate(self.subselections):
                        if subset.shape[0] != 0:
                            non_empty_sets.append(True)
                            all_pos_subset = np.sum(self.data[subset], axis=0)
                            if cur_metric_var == "Intra feature fraction":
                                heatmap_data[:,
                                             subset_idx] = all_pos_subset/self.data[subset].shape[0]
                            elif cur_metric_var == "Intra cluster fraction":
                                heatmap_data[:,
                                             subset_idx] = all_pos_subset/all_pos
                            elif cur_metric_var == "Total fraction":
                                heatmap_data[:,
                                             subset_idx] = all_pos_subset/self.data.shape[0]
                        else:
                            non_empty_sets.append(False)
                            heatmap_data[:, subset_idx] = np.zeros(
                                len(self.feature_selection))

                    # heatmap_data = heatmap_data[self.feature_selection]
                    heatmap_data = heatmap_data[:, non_empty_sets]

                    y_tick_labels = np.array(col_names)[self.feature_selection]
                    x_tick_labels = np.array([subselection_var.get()
                                              for subselection_var in self.subset_names])
                    x_tick_labels = x_tick_labels[non_empty_sets]

                    if self.initial_loop == False:
                        self.axs[subplot_idx].collections[-1].colorbar.remove()

                    sns.heatmap(data=heatmap_data,
                                ax=self.axs[subplot_idx],
                                yticklabels=y_tick_labels,
                                xticklabels=x_tick_labels)

                    plot_dict = {"type": "heatmap",
                                 "subtype": "heatmap",
                                 "subplot_idx": subplot_idx,
                                 "ax": self.axs[subplot_idx],
                                 "data": self.data,
                                 "feature_selection": self.feature_selection,
                                 "col_names": col_names,
                                 "half_range": half_range,
                                 "subset_names": self.subset_names}
                    self.plot_dicts[subplot_idx] = plot_dict

            for plot_dict in self.plot_dicts:
                if "draggable_annot" in plot_dict:
                    plot_dict["draggable_annot"].blend_out()

            fig.canvas.draw()

            for plot_dict in self.plot_dicts:
                if "draggable_annot" in plot_dict:
                    plot_dict["draggable_annot"].get_blit()
                    plot_dict["draggable_annot"].blend_in()
                if "selector" in plot_dict:
                    plot_dict["selector"].get_blit()

            fig.canvas.draw()
            self.last_frame = frame

            def wait(self):
                var = tk.IntVar()
                self.after(
                    int(float(self.fps_variable.get()) * 1000), var.set, 1)
                self.initial_loop = False
                self.wait_variable(var)

            if self.animation_switch.get() == 1:
                wait(self)
                self.initial_loop = False
                for subplot_idx, _ in enumerate(self.plot_objects):
                    if "selector" in self.plot_dicts[subplot_idx]:
                        self.plot_dicts[subplot_idx]["selector"].disconnect()
                    if self.frame_vars[subplot_idx].get() != "":
                        next_frame = int(
                            self.frame_vars[subplot_idx].get()) + 1
                        self.frame_vars[subplot_idx].set(str(next_frame))

            else:
                fig.canvas.mpl_connect("key_press_event", accept)
                self.initial_loop = False
                self.wait_variable(self.pause_var)

        self.destroy()

from checkbox_events import feature_checkbox_event, subselection_checkbox_event
from two_d_tour import launch_2d_tour
from one_d_tour import launch_1d_tour
from scatterplot import launch_scatterplot
from histogram import launch_histogram
from cat_clust_interface import launch_cat_clust_interface
from mosaic import launch_mosaic
from heatmap import launch_heatmap
import tkinter as tk
from functools import partial
from datetime import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk


def interactive_tour(data, col_names, plot_objects, half_range=None, n_max_cols=None,
                     preselection=None, preselection_names=None, n_subsets=3, size=5):
    """Launch InteractiveTourInterface object"""

    # The suicide argument causes the window to close after inital plotting
    # Restarting the app massively increases performance
    # It is unknown why
    app = InteractiveTourInterface(data,
                                   col_names,
                                   plot_objects,
                                   half_range,
                                   n_max_cols,
                                   preselection,
                                   preselection_names,
                                   n_subsets,
                                   size,
                                   suicide=True)
    app.mainloop()

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
                 preselection, preselection_names, n_subsets, size, suicide=False):
        super().__init__()

        def accept(event):
            if event.key == "right" or event.key == "left":
                self.initial_loop = False

                for subplot_idx, _ in enumerate(self.plot_objects):

                    # if "selector" in self.plot_dicts[subplot_idx]:
                    #    self.plot_dicts[subplot_idx]["selector"].disconnect()

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
                self.frame_update = True
                self.pause_var.set(0)

        self.title("Interactive tourr")
        self.data = data
        self.col_names = col_names
        self.half_range = half_range
        self.plot_objects = plot_objects
        self.displayed_tour = "Original tour"
        self.r = r
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
            self.fig, self.axs = plt.subplots(
                1, 1, figsize=(self.size, self.size))
            self.axs = [self.axs]
        else:
            n_plots = len(plot_objects)
            if n_max_cols is None:
                n_max_cols = 3
            n_cols = int(min(n_max_cols, n_plots))
            n_rows = int((n_plots + n_cols - 1) // n_cols)

            self.fig, self.axs = plt.subplots(
                n_rows, n_cols, figsize=(self.size*n_cols,
                                         self.size*n_rows),
                layout="compressed")
            self.axs = self.axs.flatten()
            for i in range(n_plots, len(self.axs)):
                self.fig.delaxes(self.axs[i])
            self.axs = self.axs[:n_plots]

        canvas = FigureCanvasTkAgg(self.fig, self)
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
                self.plot_dicts[subplot_idx]["reset_selection_check"] = True

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
                                 text=f"Plot #{subplot_idx+1} Frame")
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
            self.fig.savefig(f"{save_dir}/{now}/figure.png")
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
                      "Guided tour - LDA"]

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

                    elif self.selected_tour_type.get() == "Guided tour - LDA":
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
            self.pause_var.set(0)
        self.protocol("WM_DELETE_WINDOW", cleanup)

        self.plot_dicts = [{} for i, _ in enumerate(plot_objects)]
        self.initial_loop = True

        ###### Plot loop ######
        self.blit = 0
        self.last_frame = -1
        self.frame_update = True
        while self.frame < self.n_frames:
            self.pause_var = tk.StringVar(value=42)
            for subplot_idx, plot_object in enumerate(plot_objects):
                frame = self.frame
                if self.frame_update is False or self.initial_loop is True:
                    if plot_object["type"] == "2d_tour":
                        launch_2d_tour(self, plot_object, subplot_idx)
                    elif plot_object["type"] == "1d_tour":
                        launch_1d_tour(self, plot_object, subplot_idx)
                    elif plot_object["type"] == "scatter":
                        launch_scatterplot(self, plot_object, subplot_idx)
                    elif plot_object["type"] == "hist":
                        launch_histogram(self, plot_object, subplot_idx)
                    elif plot_object["type"] == "cat_clust_interface":
                        launch_cat_clust_interface(
                            self, plot_object, subplot_idx)
                    elif plot_object["type"] == "mosaic":
                        launch_mosaic(self, plot_object, subplot_idx)
                    elif plot_object["type"] == "heatmap":
                        launch_heatmap(self, plot_object, subplot_idx)
                else:
                    if plot_object["type"] == "2d_tour":
                        self.plot_dicts[subplot_idx]["draggable_annot"].update(plot_object,
                                                                               frame)
                    elif plot_object["type"] == "1d_tour":
                        self.plot_dicts[subplot_idx]["draggable_annot"].update(plot_object,
                                                                               frame)

            if self.frame_update is False or self.initial_loop is True:
                for plot_dict in self.plot_dicts:
                    if "draggable_annot" in plot_dict:
                        plot_dict["draggable_annot"].blend_out()

                self.fig.canvas.draw()

                for plot_dict in self.plot_dicts:
                    if "draggable_annot" in plot_dict:
                        plot_dict["draggable_annot"].get_blit()
                        plot_dict["draggable_annot"].blend_in()
                    if "selector" in plot_dict:
                        plot_dict["selector"].get_blit()

                self.fig.canvas.draw()

            self.last_frame = frame
            self.frame_update = False

            if suicide == True:
                self.event_generate("<<WM_DELETE_WINDOW>>")
                break

            def wait(self):
                var = tk.IntVar()
                self.after(
                    int(float(self.fps_variable.get()) * 1000), var.set, 1)
                self.initial_loop = False
                self.wait_variable(var)

            if self.animation_switch.get() == 1:
                wait(self)
                self.initial_loop = False
                self.frame_update = True
                for subplot_idx, _ in enumerate(self.plot_objects):
                    if "selector" in self.plot_dicts[subplot_idx]:
                        self.plot_dicts[subplot_idx]["selector"].disconnect()
                    if self.frame_vars[subplot_idx].get() != "":
                        next_frame = int(
                            self.frame_vars[subplot_idx].get()) + 1
                        self.frame_vars[subplot_idx].set(str(next_frame))

            else:
                self.fig.canvas.mpl_connect("key_press_event", accept)
                self.initial_loop = False
                self.wait_variable(self.pause_var)

        self.destroy()

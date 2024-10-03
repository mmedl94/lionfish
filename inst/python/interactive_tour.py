import tkinter as tk
from functools import partial
from datetime import datetime
import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk

# Import the required functions for various plot types
from checkbox_events import feature_checkbox_event, subselection_checkbox_event
from two_d_tour import launch_2d_tour
from one_d_tour import launch_1d_tour
from scatterplot import launch_scatterplot
from histogram import launch_histogram
from cat_clust_interface import launch_cat_clust_interface
from mosaic import launch_mosaic
from heatmap import launch_heatmap


def load_interactive_tour(data, directory_to_save, feature_names, half_range=None,
                          n_plot_cols=None, preselection=None,
                          preselection_names=None, n_subsets=None, display_size=5,
                          hover_cutoff=10, label_size=15, axes_blendout_threshhold=1):

    with open(os.path.join(directory_to_save, "attributes.pkl"), "rb") as f:
        attributes = pkl.load(f)
    plot_objects = attributes["plot_objects"]

    if half_range == None:
        half_range = attributes["half_range"]
    if n_plot_cols == None:
        n_plot_cols = attributes["n_plot_cols"]
    if preselection == None:
        preselection = attributes["preselection"]
    if n_subsets == None or n_subsets == False:
        n_subsets = attributes["n_subsets"]
    elif n_subsets < attributes["n_subsets"]:
        n_subsets = attributes["n_subsets"]
    if display_size == None:
        display_size = attributes["display_size"]
    if hover_cutoff == None:
        hover_cutoff = attributes["hover_cutoff"]
    if label_size == None:
        label_size = attributes["label_size"]

    interactive_tour(data, plot_objects, feature_names, half_range,
                     n_plot_cols, preselection,
                     preselection_names, n_subsets, display_size,
                     hover_cutoff, label_size, load=True,
                     directory_to_save=directory_to_save,
                     axes_blendout_threshhold=axes_blendout_threshhold)


def interactive_tour(data, plot_objects, feature_names, half_range=None,
                     n_plot_cols=None, preselection=None,
                     preselection_names=None, n_subsets=3, display_size=5,
                     hover_cutoff=10, label_size=15, load=False,
                     directory_to_save=False, axes_blendout_threshhold=1):
    """Launch InteractiveTourInterface for interactive plotting."""

    if matplotlib.get_backend() != "TkAgg":
        matplotlib.use("TkAgg")

    app = InteractiveTourInterface(data, plot_objects, feature_names, half_range,
                                   n_plot_cols, preselection, preselection_names,
                                   n_subsets, display_size, hover_cutoff, label_size,
                                   load, directory_to_save, axes_blendout_threshhold)
    app.mainloop()


class InteractiveTourInterface(ctk.CTk):
    def __init__(self, data, plot_objects, feature_names, half_range=None,
                 n_plot_cols=None, preselection=None,
                 preselection_names=None, n_subsets=3, display_size=5,
                 hover_cutoff=10, label_size=15, load=False,
                 directory_to_save=False, axes_blendout_threshhold=1):
        super().__init__()

        self.title("Interactive Tour")
        self.r = r
        self.data = data
        self.n_pts = self.data.shape[0]
        self.feature_names = feature_names
        self.half_range = half_range or self.calculate_half_range(data)
        self.plot_objects = plot_objects if isinstance(
            plot_objects, list) else [plot_objects]
        self.display_size = display_size
        self.n_subsets = max(n_subsets, len(set(preselection))
                             ) if preselection else n_subsets
        self.n_subsets = int(self.n_subsets)
        self.preselection = np.array(
            preselection, dtype=int) - 1 if preselection else None
        self.preselection_names = preselection_names
        self.axes_blendout_threshhold = axes_blendout_threshhold

        self.subselections = self.initialize_subselections()
        self.orig_subselections = self.subselections.copy()
        self.colors = self.get_colors()
        self.n_bins = tk.StringVar(self, "26")

        self.setup_cleanup()
        self.setup_plot_layout(n_plot_cols)
        self.sidebar_row_tracker = 0
        self.setup_sidebar()

        self.limits = 1
        self.n_frames = 1
        self.obs_idx_ = np.arange(0, self.data.shape[0])
        self.displayed_tour = "Original tour"

        self.fc = self.initialize_color_array()
        self.hover_cutoff = hover_cutoff
        self.label_size = label_size
        self.plot_dicts = [{} for _ in self.plot_objects]
        self.frame = 0
        self.last_frame = -1
        self.frame_update = True
        self.initial_loop = True

        self.bind_all("<KeyPress>", self.accept)

        self.load = load
        self.directory_to_save = directory_to_save

        self.plot_loop()

    def calculate_half_range(self, data):
        """Calculate the default half range based on the data."""
        return np.max(np.sqrt(np.sum(data**2, axis=1)))

    def get_colors(self):
        """Get color palette for the plots."""
        colors = matplotlib.colormaps["tab10"].colors
        return [[r, g, b, 1.0] for [r, g, b] in colors]

    def initialize_subselections(self):
        """Initialize the subselections array."""
        # Placeholder logic for initializing subselections
        if self.preselection is not None:
            return [np.where(self.preselection == i)[0] for i in range(self.n_subsets)]
        else:
            return [np.arange(self.n_pts) if i == 0 else np.array([]) for i in range(self.n_subsets)]

    def setup_cleanup(self):
        """Setup cleanup method for when the window is closed."""
        def cleanup():
            self.frame = self.n_frames
            self.destroy()
            plt.close("all")
        self.protocol("WM_DELETE_WINDOW", cleanup)

    def setup_plot_layout(self, n_plot_cols):
        """Setup the layout for the plots based on the number of plots."""
        n_plots = len(self.plot_objects)
        # Ensure n_cols is an integer
        n_cols = min(int(n_plot_cols or 3), n_plots)
        # Ensure n_rows is an integer
        n_rows = int((n_plots + n_cols - 1) // n_cols)

        # Create subplots and handle the case of a single plot
        self.fig, self.axs = plt.subplots(n_rows, n_cols,
                                          figsize=(
                                              self.display_size * n_cols, self.display_size * n_rows),
                                          layout="compressed")
        if n_plots == 1:
            self.axs = [self.axs]
        else:
            self.axs = self.axs.flatten()[:n_plots]

        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().grid(row=0, column=1, sticky="n")

    def setup_sidebar(self):
        """Setup the sidebar with various control options."""
        sidebar = ctk.CTkScrollableFrame(self)
        sidebar.grid(row=0, column=0, sticky="nswe", padx=2, pady=2)

        self.setup_feature_selection(sidebar)
        self.setup_subselection(sidebar)
        self.setup_frame_controls(sidebar)
        self.setup_histogram_controls(sidebar)
        self.setup_animation_controls(sidebar)
        self.setup_blendout_projection(sidebar)
        self.setup_save_button(sidebar)
        self.setup_load_button(sidebar)
        self.setup_tour_controls(sidebar)
        self.setup_metric_menu(sidebar)

        max_width = 0
        for widget in sidebar.winfo_children():
            widget.update_idletasks()
            widget_width = widget.winfo_reqwidth()
            max_width = max(max_width, widget_width)

        # Set the width of the sidebar to the largest widget width
        sidebar.configure(width=max_width*1.25)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_propagate()

    def setup_feature_selection(self, sidebar):
        """Setup the feature selection checkboxes."""
        feature_selection_frame = ctk.CTkFrame(sidebar)
        feature_selection_frame.grid(row=self.sidebar_row_tracker,
                                     column=0,
                                     sticky="n")
        self.sidebar_row_tracker += 1

        self.feature_selection_vars = []
        self.feature_selection = []

        for feature_idx, feature in enumerate(self.feature_names):
            check_var = tk.IntVar(self, 1)
            self.feature_selection_vars.append(check_var)
            self.feature_selection.append(1)
            checkbox = ctk.CTkCheckBox(
                master=feature_selection_frame, text=feature,
                command=partial(feature_checkbox_event, self, feature_idx),
                variable=check_var, onvalue=1, offvalue=0
            )
            checkbox.grid(row=feature_idx, column=0, pady=3, sticky="w")

        self.feature_selection = np.bool_(self.feature_selection)

    def setup_subselection(self, sidebar):
        """Setup the subselection checkboxes and textboxes."""
        subselection_frame = ctk.CTkFrame(sidebar)
        # Add padding for spacing
        subselection_frame.grid(row=self.sidebar_row_tracker,
                                column=0, padx=10, pady=10, sticky="n")
        self.sidebar_row_tracker += 1

        self.subselection_vars = []
        self.subset_names = []

        for subselection_idx in range(self.n_subsets):
            # Checkbox for the subselection
            if subselection_idx == 0:
                check_var = tk.IntVar(self, 1)
            else:
                check_var = tk.IntVar(self, 0)
            self.subselection_vars.append(check_var)

            checkbox = ctk.CTkCheckBox(
                master=subselection_frame,
                text="",
                width=24,
                command=partial(
                    subselection_checkbox_event, self, subselection_idx),
                variable=check_var,
                onvalue=1,
                offvalue=0)

            checkbox.grid(row=subselection_idx,
                          column=0,
                          padx=2,
                          pady=2,
                          sticky="w")

            # Textbox for the subset name
            subset_name = self.get_subselection_name(subselection_idx)
            textvariable = tk.StringVar(self, subset_name)
            self.subset_names.append(textvariable)

            textbox = ctk.CTkEntry(
                master=subselection_frame, textvariable=textvariable)
            # Align textboxes next to the checkboxes
            textbox.grid(row=subselection_idx, column=1,
                         padx=5, pady=5, sticky="w")

            color_box = ctk.CTkButton(
                master=subselection_frame,
                text="",
                width=24,
                height=24,
                hover=False,
                fg_color=matplotlib.colors.rgb2hex(
                    self.colors[subselection_idx]),
                command=partial(self.update_colors, subselection_idx))

            color_box.grid(row=subselection_idx,
                           column=2,
                           padx=2,
                           pady=2,
                           sticky="w")

        # Add reset button below the subselections
        reset_selection_button = ctk.CTkButton(
            master=subselection_frame,
            text="Reset original selection",
            command=partial(self.reset_selection, self))

        reset_selection_button.grid(
            row=self.n_subsets,
            column=0,
            columnspan=3,
            pady=10,
            sticky="n")

    def update_colors(self, subselection_idx):
        """Update the color of the selected subset."""
        # Get the color of the current subselection
        col_array = np.array(self.colors[subselection_idx])

        # Find all rows that match this color
        selected_row_idx = np.where(
            np.all(self.fc[:, :3] == col_array[:3], axis=1))

        # Toggle the alpha value (opacity) between fully visible and semi-transparent
        if self.colors[subselection_idx][-1] == 1:
            # Make semi-transparent
            self.colors[subselection_idx][-1] = 0.3
            # Reduce opacity in the face color array
            self.fc[selected_row_idx, -1] = 0.1
        else:
            # Make fully visible
            self.colors[subselection_idx][-1] = 1
            # Set full opacity in the face color array
            self.fc[selected_row_idx, -1] = 1

        # Mark plots with tours to update them properly
        for subplot_idx, plot_dict in enumerate(self.plot_dicts):
            if plot_dict.get("subtype") in ["1d_tour", "2d_tour"]:
                self.plot_dicts[subplot_idx]["update_plot"] = False

        # Re-run the plot loop to apply color changes
        self.plot_loop()

    def get_subselection_name(self, subselection_idx):
        """Get the name for a given subselection."""
        # Ensure preselection_names is populated and check if the index is within bounds
        if self.preselection_names and subselection_idx < len(self.preselection_names):
            return self.preselection_names[subselection_idx]
        else:
            return f"Subset {subselection_idx + 1}"

    def initialize_color_array(self):
        """Initialize the color array based on the selections."""
        fc = np.repeat(np.array(self.colors[0])[
                       :, np.newaxis], self.data.shape[0], axis=1).T
        for idx, subset in enumerate(self.subselections):
            if subset.size:
                fc[subset] = self.colors[idx]
        return fc

    def setup_frame_controls(self, sidebar):
        """Setup controls for frame selection."""
        frame_selection_frame = ctk.CTkFrame(sidebar)
        frame_selection_frame.grid(row=self.sidebar_row_tracker, column=0)
        self.sidebar_row_tracker += 1
        self.frame_vars = []
        self.frame_textboxes = []

        for subplot_idx, plot_object in enumerate(self.plot_objects):
            self.plot_objects[subplot_idx]["og_obj"] = self.plot_objects[subplot_idx]["obj"]
            # Create a label for the plot frame
            label = ctk.CTkLabel(master=frame_selection_frame,
                                 text=f"Plot #{subplot_idx + 1} Frame")
            label.grid(row=subplot_idx, column=0, pady=3, padx=0, sticky="w")

            # Initialize textvariable for the entry widget
            textvariable = tk.StringVar(self, "")

            # Create a disabled entry widget for the frame variable
            textbox = ctk.CTkEntry(
                master=frame_selection_frame, textvariable=textvariable, width=40, state="disabled", fg_color="grey")
            textbox.grid(row=subplot_idx, column=1, pady=3, padx=0, sticky="w")

            # Store the textvariable and textbox for future reference
            self.frame_vars.append(textvariable)
            self.frame_textboxes.append(textbox)

        # Button to update frames
        update_button = ctk.CTkButton(
            master=frame_selection_frame,
            text="Update",
            command=self.plot_loop
        )
        update_button.grid(
            row=len(self.plot_objects), column=0, columnspan=2, pady=(3, 3), sticky="n")

    def setup_histogram_controls(self, sidebar):
        plot_types_w_hist = ["histogram", "1d_tour"]

        need_hist_control = any(
            plot_object["type"] in plot_types_w_hist for plot_object in self.plot_objects)

        if need_hist_control is False:
            return

        histogram_controls_frame = ctk.CTkFrame(sidebar)
        histogram_controls_frame.grid(row=self.sidebar_row_tracker, column=0)
        self.sidebar_row_tracker += 1

        label = ctk.CTkLabel(master=histogram_controls_frame,
                             text="Number of bins of histograms")
        label.grid(row=0, column=0,
                   pady=3, padx=0, sticky="w")

        textbox = ctk.CTkEntry(
            master=histogram_controls_frame,
            textvariable=self.n_bins, width=40)
        textbox.grid(row=0, column=1,
                     pady=3, padx=0, sticky="w")

    def setup_animation_controls(self, sidebar):
        """Setup animation control widgets."""
        animation_frame = ctk.CTkFrame(sidebar)
        animation_frame.grid(row=self.sidebar_row_tracker, column=0)
        self.sidebar_row_tracker += 1

        label = ctk.CTkLabel(master=animation_frame,
                             text="Animate")
        label.grid(row=0, column=0,
                   pady=3, padx=3, sticky="w")

        self.animation_switch = tk.IntVar(self, 0)
        animation_checkbox = ctk.CTkCheckBox(
            master=animation_frame,
            text="", width=24,
            variable=self.animation_switch,
            command=self.plot_loop,
            onvalue=1, offvalue=0
        )
        animation_checkbox.grid(row=0, column=1, pady=3)

        self.fps_variable = tk.StringVar(self, "1")
        fps_textbox = ctk.CTkEntry(
            master=animation_frame, width=40, textvariable=self.fps_variable)
        fps_textbox.grid(row=0, column=2, pady=3)

        label = ctk.CTkLabel(master=animation_frame, text="seconds")
        label.grid(row=0, column=3, pady=3)

    def setup_blendout_projection(self, sidebar):
        blendout_projection_frame = ctk.CTkFrame(sidebar)
        blendout_projection_frame.grid(row=self.sidebar_row_tracker, column=0)
        self.sidebar_row_tracker += 1

        self.blendout_projection_switch = tk.IntVar(self, 0)
        blendout_projection_checkbox = ctk.CTkCheckBox(
            master=blendout_projection_frame,
            text="", width=24,
            variable=self.blendout_projection_switch,
            command=self.blendout_event,
            onvalue=1, offvalue=0
        )
        blendout_projection_checkbox.grid(row=0, column=0, pady=3)

        label = ctk.CTkLabel(master=blendout_projection_frame,
                             text="Blend out projection threshold")
        label.grid(row=0, column=1, pady=3)

        self.blendout_projection_variable = tk.StringVar(self,
                                                         str(self.axes_blendout_threshhold))
        blendout_projection_textbox = ctk.CTkEntry(
            master=blendout_projection_frame, width=40,
            textvariable=self.blendout_projection_variable)
        blendout_projection_textbox.grid(row=0, column=2, pady=3)

    def blendout_event(self, event=None):
        for plot_idx, _ in enumerate(self.plot_dicts):
            self.plot_dicts[plot_idx]["update_plot"] = False
        self.plot_loop()

    def setup_save_button(self, sidebar):
        """Setup the save button for saving projections and subsets."""
        save_button = ctk.CTkButton(
            master=sidebar, width=100, height=32, border_width=0, corner_radius=8,
            text="Save projections \n and subsets", command=partial(self.save_event)
        )
        save_button.grid(row=self.sidebar_row_tracker,
                         column=0, pady=(3, 3), sticky="n")
        self.sidebar_row_tracker += 1

    def setup_load_button(self, sidebar):
        """Setup the load button for recovering a saved state."""
        load_button = ctk.CTkButton(
            master=sidebar, width=100, height=32, border_width=0, corner_radius=8,
            text="Load projections \n and subsets", command=partial(self.load_event)
        )
        load_button.grid(row=self.sidebar_row_tracker,
                         column=0, pady=(3, 3), sticky="n")
        self.sidebar_row_tracker += 1

    def load_event(self):
        if self.load is True:
            load_dir = self.directory_to_save
        else:
            load_dir = ctk.filedialog.askdirectory()

        with open(os.path.join(load_dir, "attributes.pkl"), "rb") as f:
            attributes = pkl.load(f)

        # drop these attributes if we load from scratch

        attributes_to_drop = ["half_range",
                              "preselection",
                              "n_subsets",
                              "display_size",
                              "hover_cutoff",
                              "label_size",
                              "initial_loop"]

        for attribute_to_drop in attributes_to_drop:
            del attributes[attribute_to_drop]

        self.__dict__.update(attributes)

        def set_tk_states(saved_vars, var_list):
            for idx, value in enumerate(saved_vars):
                var_list[idx].set(value)

        set_tk_states(attributes["frame_vars_"], self.frame_vars)
        set_tk_states(attributes["feature_selection_vars_"],
                      self.feature_selection_vars)
        while len(attributes["subselection_vars_"]) < len(self.subselection_vars):
            attributes["subselection_vars_"].append(0)
            attributes["subselections"].append(np.array([], dtype=int))
        set_tk_states(attributes["subselection_vars_"], self.subselection_vars)
        if "metric_vars_" in attributes:
            set_tk_states(attributes["metric_vars_"], self.metric_vars)

        self.plot_dicts_ = [{} for _ in self.plot_objects]
        # restore projections
        projections = attributes["projections"]
        for idx, plot_dict in enumerate(self.plot_dicts):
            if str(idx) in projections:
                self.plot_dicts[idx]["proj"] = projections[str(idx)]
                if self.load is True:
                    self.plot_dicts_[idx]["proj"] = projections[str(idx)]

        with open(os.path.join(load_dir, "tkinter_states.pkl"), "rb") as f:
            tkinter_states = pkl.load(f)

        for var_name, value in tkinter_states.items():
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                if isinstance(var, (tk.IntVar, tk.StringVar, tk.BooleanVar, tk.DoubleVar)):
                    var.set(value)

        for subplot_idx in range(len(self.plot_objects)):
            self.plot_dicts[subplot_idx]["reset_selection_check"] = True

        if self.load is True:
            self.load = False

        self.plot_loop()

    def is_picklable(self, obj):
        """Check if attribute is picklable"""
        try:
            pkl.dumps(obj)  # Attempt to pickle the object
            return True
        except (pkl.PicklingError, TypeError):  # Catch exceptions related to pickling
            return False

    def save_event(self):
        """Handle saving projections and subsets to files."""
        save_dir = ctk.filedialog.askdirectory()
        now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        save_path = os.path.join(save_dir, now)

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # Save the subsets
        save_df = pd.DataFrame(self.subselections, dtype=pd.Int64Dtype()).T
        save_df = save_df + 1
        save_df.columns = [subset_name.get()
                           for subset_name in self.subset_names]
        restructured_data = []
        for subset in save_df.columns:
            subset_idx = subset.split(" ")[1]
            for observation in save_df[subset].dropna():
                restructured_data.append([subset_idx, observation])
        save_df = pd.DataFrame(restructured_data, columns=[
                               "subset", "observation_index"])
        save_df.to_csv(os.path.join(
            save_path, "subset_selection.csv"), index=False)

        # Save attributes
        attributes_to_save = {
            k: v for k, v in self.__dict__.items()
            if k != 'r' and self.is_picklable(v)
        }
        # Drop attributes that should not be saved
        attributes_to_drop = ["fig",
                              "axs",
                              "data",
                              "load",
                              "_CTkAppearanceModeBaseClass__appearance_mode",
                              "_CTkScalingBaseClass__scaling_type",
                              "_tclCommands",
                              "master",
                              "_tkloaded",
                              "_CTkScalingBaseClass__window_scaling",
                              "_current_width",
                              "_current_height",
                              "_min_width",
                              "_min_height",
                              "_max_width",
                              "_max_height",
                              "_last_resizable_args",
                              "_iconbitmap_method_called",
                              "_state_before_windows_set_titlebar_color",
                              "_window_exists",
                              "_withdraw_called_before_window_exists",
                              "_iconify_called_before_window_exists",
                              "_block_update_dimensions_event",
                              "focused_widget_before_widthdraw"]

        for attribute in attributes_to_drop:
            del attributes_to_save[attribute]

        projections = {}
        for idx, plot_dict in enumerate(self.plot_dicts):
            if "proj" in plot_dict:
                projections[str(idx)] = plot_dict["proj"]

        def get_tk_states(var_list):
            return [var.get() for var in var_list]
        try:
            attributes_to_save.update({
                "frame_vars_": get_tk_states(self.frame_vars),
                "feature_selection_vars_": get_tk_states(self.feature_selection_vars),
                "subselection_vars_": get_tk_states(self.subselection_vars),
                "metric_vars_": get_tk_states(self.metric_vars),
                "projections": projections
            })
        except AttributeError:
            attributes_to_save.update({
                "frame_vars_": get_tk_states(self.frame_vars),
                "feature_selection_vars_": get_tk_states(self.feature_selection_vars),
                "subselection_vars_": get_tk_states(self.subselection_vars),
                "projections": projections
            })

        with open(os.path.join(save_path, "attributes.pkl"), "wb") as f:
            pkl.dump(attributes_to_save, f)

        # save additional tkinter states
        tkinter_states = {}
        for var_name, var in self.__dict__.items():
            if isinstance(var, (tk.IntVar, tk.StringVar, tk.BooleanVar, tk.DoubleVar)):
                tkinter_states[var_name] = var.get()
        with open(os.path.join(save_path, "tkinter_states.pkl"), "wb") as f:
            pkl.dump(tkinter_states, f)

        # save feature selection
        features = np.array([self.feature_names, self.feature_selection*1]).T
        feature_df = pd.DataFrame(features, columns=["features", "selected"])
        feature_df.to_csv(os.path.join(save_path, "feature_selection.csv"),
                          index=False,
                          header=False)

        # Save the figure
        self.fig.savefig(os.path.join(save_path, "figure.png"),
                         dpi=300)

        # Save projections
        for idx, plot_dict in enumerate(self.plot_dicts):
            if "proj" in plot_dict:
                proj_df = pd.DataFrame(
                    plot_dict["proj"][self.feature_selection])
                proj_df["original variables"] = np.array(self.feature_names)[
                    self.feature_selection]
                proj_df.set_index("original variables", inplace=True)
                proj_df.to_csv(os.path.join(
                    save_path, f"projection_object_{idx + 1}.csv"))

        # prevent projections from resetting
        for subplot_idx, plot_dict in enumerate(self.plot_dicts):
            if plot_dict.get("subtype") in ["1d_tour", "2d_tour"]:
                self.plot_dicts[subplot_idx]["update_plot"] = False

        self.plot_loop()

    def setup_tour_controls(self, sidebar):
        """Setup the controls for running and resetting tours."""
        tour_types = ["Local tour", "Guided tour - holes",
                      "Guided tour - holes - better", "Guided tour - LDA"]

        self.selected_tour_type = ctk.StringVar(value="Local tour")
        tour_menu = ctk.CTkComboBox(
            master=sidebar, values=tour_types, variable=self.selected_tour_type)
        tour_menu.grid(row=self.sidebar_row_tracker,
                       column=0, pady=(3, 3), sticky="n")
        self.sidebar_row_tracker += 1

        run_tour_button = ctk.CTkButton(
            master=sidebar, text="Run tour", command=partial(self.run_tour)
        )
        run_tour_button.grid(row=self.sidebar_row_tracker,
                             column=0, pady=(3, 3), sticky="n")
        self.sidebar_row_tracker += 1

        reset_tour_button = ctk.CTkButton(
            master=sidebar, text="Reset original tour", command=partial(self.reset_original_tour)
        )
        reset_tour_button.grid(row=self.sidebar_row_tracker,
                               column=0, pady=(3, 3), sticky="n")
        self.sidebar_row_tracker += 1

    def run_tour(self):
        """Run a new tour based on the selected tour type."""
        for idx, plot_object in enumerate(self.plot_objects):
            if plot_object["type"] in ["1d_tour", "2d_tour"]:
                dimension = 1 if plot_object["type"] == "1d_tour" else 2
                full_array = np.zeros(
                    (self.feature_selection.shape[0], dimension))

                if self.selected_tour_type.get() == "Local tour":
                    new_proj = self.r.get_local_history(
                        self.data[:2, self.feature_selection],
                        self.plot_dicts[idx]["proj"][self.feature_selection]
                    )
                elif self.selected_tour_type.get() == "Guided tour - holes":
                    new_proj = self.r.get_guided_holes_history(
                        self.data[:, self.feature_selection], dimension)
                elif self.selected_tour_type.get() == "Guided tour - holes - better":
                    new_proj = self.r.get_guided_holes_better_history(
                        self.data[:, self.feature_selection], dimension)
                elif self.selected_tour_type.get() == "Guided tour - LDA":
                    subselection_idxs = np.zeros(self.data.shape[0], dtype=int)
                    for subselection_idx, arr in enumerate(self.subselections):
                        if arr.size:
                            subselection_idxs[arr] = subselection_idx + 1

                    new_proj = self.r.get_guided_lda_history(self.data[:, self.feature_selection],
                                                             subselection_idxs, dimension)

                full_array = np.tile(
                    full_array[:, :, np.newaxis], (1, 1, new_proj.shape[2]))
                full_array[self.feature_selection] = new_proj

                self.displayed_tour = self.selected_tour_type.get()
                self.plot_objects[idx]["og_frame"] = int(
                    self.frame_vars[idx].get())
                self.plot_objects[idx]["obj"] = full_array
                self.frame_vars[idx].set("0")

        self.plot_loop()

    def reset_original_tour(self):
        """Reset to the original tour."""
        for idx, plot_object in enumerate(self.plot_objects):
            if plot_object["type"] in ["1d_tour", "2d_tour"]:
                self.plot_objects[idx]["obj"] = plot_object["og_obj"]
                self.frame_vars[idx].set(
                    str(self.plot_objects[idx]["og_frame"]))
                self.displayed_tour = "Original tour"

        self.plot_loop()

    def setup_metric_menu(self, sidebar):
        """Setup the metric selection menu if needed."""
        metrics = ["Intra cluster fraction",
                   "Intra feature fraction",
                   "Total fraction"]
        plot_types_w_metric = ["heatmap", "cat_clust_interface"]

        # Check if any plot requires a metric selection
        need_metric = any(
            plot_object["type"] in plot_types_w_metric for plot_object in self.plot_objects)

        if need_metric is False:
            return

        metric_selection_frame = ctk.CTkFrame(sidebar)
        metric_selection_frame.grid(row=self.sidebar_row_tracker, column=0)
        self.sidebar_row_tracker += 1
        self.metric_vars = []

        for subplot_idx, plot_object in enumerate(self.plot_objects):
            metric_var = tk.StringVar()

            if plot_object["type"] in plot_types_w_metric:
                metric_var.set(plot_object.get(
                    "obj", "Intra cluster fraction of positive"))
            else:
                metric_var.set(metrics[0])

            label = ctk.CTkLabel(
                master=metric_selection_frame, text=f"Plot #{subplot_idx + 1}")
            label.grid(row=subplot_idx, column=0, pady=(3, 3), sticky="n")

            metric_selection_menu = ctk.CTkComboBox(
                master=metric_selection_frame,
                values=metrics,
                command=self.metric_selection_event,
                variable=metric_var
            )

            metric_selection_menu.grid(
                row=subplot_idx, column=1, pady=(3, 3), sticky="n")

            if plot_object["type"] not in plot_types_w_metric:
                metric_selection_menu.configure(
                    state="disabled",
                    fg_color="grey")

            self.metric_vars.append(metric_var)

    def metric_selection_event(self, event=None):
        for plot_idx, _ in enumerate(self.plot_dicts):
            self.plot_dicts[plot_idx]["update_plot"] = False
        self.plot_loop()

    def plot_loop(self, event=None):
        """Main loop to handle plotting and animation."""
        for subplot_idx, plot_object in enumerate(self.plot_objects):
            if not self.frame_update or self.initial_loop:
                self.launch_plot(subplot_idx, plot_object)
            else:
                self.update_plot(subplot_idx, plot_object)

        if not self.frame_update or self.initial_loop:
            self.draw_plots()

        self.last_frame = self.frame
        self.frame_update = False

        if self.animation_switch.get() == 1:
            self.animate_plots()
        else:
            self.initial_loop = False

        if self.load is True:
            self.load_event()

    def launch_plot(self, subplot_idx, plot_object):
        """Launch the appropriate plot based on the plot type."""
        plot_type = plot_object["type"]
        if plot_type == "2d_tour":
            launch_2d_tour(self, plot_object, subplot_idx)
        elif plot_type == "1d_tour":
            launch_1d_tour(self, plot_object, subplot_idx)
        elif plot_type == "scatter":
            launch_scatterplot(self, plot_object, subplot_idx)
        elif plot_type == "hist":
            launch_histogram(self, plot_object, subplot_idx)
        elif plot_type == "cat_clust_interface":
            launch_cat_clust_interface(self, plot_object, subplot_idx)
        elif plot_type == "mosaic":
            launch_mosaic(self, plot_object, subplot_idx)
        elif plot_type == "heatmap":
            launch_heatmap(self, plot_object, subplot_idx)

    def update_plot(self, subplot_idx, plot_object):
        """Update the plot for the current frame."""
        plot_type = plot_object["type"]
        if plot_type in ["2d_tour", "1d_tour"]:
            self.plot_dicts[subplot_idx]["draggable_annot"].update(
                plot_object, self.frame)

    def draw_plots(self):
        """Redraw the plots on the canvas."""
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

    def animate_plots(self):
        """Animate the plots by updating the frames."""
        self.initial_loop = False
        self.frame_update = True

        for subplot_idx in range(len(self.plot_objects)):
            if self.frame_vars[subplot_idx].get() != "":
                next_frame = int(self.frame_vars[subplot_idx].get()) + 1
                self.frame_vars[subplot_idx].set(str(next_frame))

        self.after(int(float(self.fps_variable.get()) * 1000), self.plot_loop)

    def reset_selection(self, event=None):
        """Reset to the original selection of subsets."""
        self.subselections = self.orig_subselections.copy()
        self.colors = self.get_colors()
        self.fc = self.original_fc.copy()

        for subplot_idx in range(len(self.plot_objects)):
            if "selector" in self.plot_dicts[subplot_idx]:
                self.plot_dicts[subplot_idx]["selector"].disconnect()

            self.plot_dicts[subplot_idx]["reset_selection_check"] = True

        self.plot_loop()

    def accept(self, event):
        """Handle key press events for navigation."""
        key = event.keysym
        if key in ["Right", "Left"]:
            self.initial_loop = False

            for subplot_idx in range(len(self.plot_objects)):
                if self.frame_vars[subplot_idx].get() != "":
                    current_frame = int(self.frame_vars[subplot_idx].get())
                    next_frame = current_frame + \
                        1 if key == "Right" else max(current_frame - 1, 0)
                    self.frame_vars[subplot_idx].set(str(next_frame))

            self.frame_update = True
            self.plot_loop()

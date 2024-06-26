import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from functools import partial

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helpers import gram_schmidt
# Helper class that manages the lasso selection


class LassoSelect:
    def __init__(self, plot_dicts, subplot_idx, colors, n_pts, alpha_other=0.3):
        # initialize arguments
        self.n_pts = n_pts
        self.plot_dicts = plot_dicts
        self.subplot_idx = subplot_idx
        self.canvas = plot_dicts[subplot_idx]["ax"].figure.canvas
        self.collection = plot_dicts[subplot_idx]["ax"].collections[0]
        self.fc = self.plot_dicts[0]["fc"]
        self.colors = colors
        self.alpha_other = alpha_other

        # initialize lasso selector
        self.lasso = LassoSelector(
            plot_dicts[subplot_idx]["ax"],
            onselect=partial(self.onselect),
            button=1)
        self.ind = []
    # onselect governs what happens with selected data points
    # changes alpha of selected data points
    # saves indices of selected data points

    def onselect(self, verts):
        path = Path(verts)
        xys = self.collection.get_offsets()
        self.ind = np.nonzero(path.contains_points(xys))[0]

        # Check which subset is active
        for col_idx, subselection_var in enumerate(self.plot_dicts[0]["subselection_vars"]):
            # If subset is active
            if subselection_var.get() == 1:
                # change facecolors
                self.plot_dicts[0]["fc"][self.ind] = [self.colors[col_idx]]
                # Change subselections
                for idx, subselection in enumerate(self.plot_dicts[0]["subselections"]):
                    # if the looped over subset isn't the currently selected, remove
                    # newly selected indices from old subsets
                    if col_idx != idx:
                        if set(self.ind) & set(subselection):
                            updated_ind = np.setdiff1d(subselection, self.ind)
                            self.plot_dicts[0]["subselections"][idx] = updated_ind

                # get set of old selection and new selection
                selected_set = list(set(self.ind).union(
                    set(self.plot_dicts[0]["subselections"][col_idx])))
                self.plot_dicts[0]["subselections"][col_idx] = np.array(
                    selected_set)

        self.collection.set_facecolors(self.plot_dicts[0]["fc"])

        # update other plots if applicable
        for plot_dict in self.plot_dicts:
            # check plots if they are scatterplots. if so recolor datapoints
            if plot_dict["type"] == "scatter":
                collection_subplot = plot_dict["ax"].collections[0]
                collection_subplot.set_facecolors(self.plot_dicts[0]["fc"])

            elif plot_dict["type"] == "hist":
                if plot_dict["subtype"] == "1d_tour":
                    feature_selection = plot_dict["feature_selection"]
                    plot_dict["proj"][feature_selection, 0] = plot_dict["proj"][feature_selection, 0] / \
                        np.linalg.norm(
                            plot_dict["proj"][feature_selection, 0])
                    x = np.matmul(plot_dict["data"][:, feature_selection],
                                  plot_dict["proj"][feature_selection])/plot_dict["half_range"]
                    x = x[:, 0]
                else:
                    x = plot_dict["data"][:, plot_dict["hist_feature"]]

                # Get x and y_lims of old plot
                x_lims = plot_dict["ax"].get_xlim()
                y_lims = plot_dict["ax"].get_ylim()
                title = plot_dict["ax"].get_title()
                x_label = plot_dict["ax"].get_xlabel()

                x_subselections = []
                for subselection in self.plot_dicts[0]["subselections"]:
                    if subselection.shape[0] != 0:
                        x_subselections.append(x[subselection])
                    else:
                        x_subselections.append(np.array([]))

                plot_dict["ax"].clear()
                plot_dict["ax"].hist(
                    x_subselections,
                    stacked=True,
                    picker=True,
                    color=self.colors[:len(x_subselections)])

                plot_dict["ax"].set_ylim(y_lims)
                plot_dict["ax"].set_xlim(x_lims)
                plot_dict["ax"].set_title(title)
                plot_dict["ax"].set_xlabel(x_label)

        self.canvas.draw_idle()

    # governs what happens when disconnected (after pressing "enter")
    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


class BarSelect:
    def __init__(self, plot_dicts, subplot_idx, feature_selection, colors, half_range, alpha_other=0.3):
        # initialize parameters
        self.plot_dicts = plot_dicts
        self.subplot_idx = subplot_idx
        self.plot_dict = self.plot_dicts[self.subplot_idx]
        self.feature_selection = feature_selection
        self.half_range = half_range
        self.ax = plot_dicts[subplot_idx]["ax"]
        self.data = plot_dicts[subplot_idx]["data"]
        self.subtype = plot_dicts[subplot_idx]["subtype"]
        if self.subtype == "hist":
            self.hist_feature = plot_dicts[subplot_idx]["hist_feature"]

        self.canvas = self.ax.figure.canvas
        self.collection = self.ax.collections
        self.alpha_other = alpha_other
        self.patches = self.ax.patches
        self.y_lims = self.ax.get_ylim()
        self.colors = colors

        self.connection = self.ax.figure.canvas.mpl_connect('pick_event', partial(
            self.onselect))
        self.ind = []

        # transform x if necessary and save transform. Do we need this???
        for subplot_idx, plot_dict in enumerate(self.plot_dicts):
            if not isinstance(plot_dict, int):
                if plot_dict["subtype"] == "1d_tour":
                    plot_dict["proj"][self.feature_selection, 0] = plot_dict["proj"][self.feature_selection, 0] / \
                        np.linalg.norm(
                            plot_dict["proj"][self.feature_selection, 0])
                    x = np.matmul(plot_dict["data"][:, self.feature_selection],
                                  plot_dict["proj"][self.feature_selection])/self.half_range
                    self.plot_dicts[subplot_idx]["x"] = x[:, 0]
                elif plot_dict["subtype"] == "hist":
                    self.plot_dicts[subplot_idx]["x"] = plot_dict["data"][:,
                                                                          plot_dict["hist_feature"]]

    # onselect governs what happens with selected data points
    # changes alpha of selected data points
    # saves indices of selected data points

    def onselect(self, event):
        if event.artist.axes != self.ax:
            return

        min_select = event.artist.get_x()
        max_select = min_select+event.artist.get_width()

        if self.plot_dict["subtype"] == "1d_tour":
            self.plot_dict["proj"][self.feature_selection, 0] = self.plot_dict["proj"][self.feature_selection, 0] / \
                np.linalg.norm(
                    self.plot_dict["proj"][self.feature_selection, 0])
            x = np.matmul(self.plot_dict["data"][:, self.feature_selection],
                          self.plot_dict["proj"][self.feature_selection])/self.half_range
            x = x[:, 0]
            self.plot_dicts[self.subplot_idx]["x"] = x

        new_ind = np.where(np.logical_and(
            self.plot_dict["x"] >= min_select, self.plot_dict["x"] <= max_select))[0].tolist()

        # Check which subset is active
        for col_idx, subselection_var in enumerate(self.plot_dicts[0]["subselection_vars"]):
            # If subset is active
            if subselection_var.get() == 1:
                # add new_ind to old selection
                merged_selection = list(
                    self.plot_dicts[0]["subselections"][col_idx]) + list(new_ind)
                merged_selection = np.array(list(set(merged_selection)))
                self.plot_dicts[0]["subselections"][col_idx] = merged_selection

                # remove new selection from other selections
                for idx, subselection in enumerate(self.plot_dicts[0]["subselections"]):
                    # if the looped over subset isn't the currently selected, remove
                    # newly selected indices from old subsets
                    if col_idx != idx:
                        removed_selection = np.setdiff1d(
                            subselection, new_ind)
                        self.plot_dicts[0]["subselections"][idx] = removed_selection

        for col_idx, subselection in enumerate(self.plot_dicts[0]["subselections"]):
            if subselection.shape[0] != 0:
                self.plot_dicts[0]["fc"][subselection] = self.colors[col_idx]

        for plot_dict in self.plot_dicts:
            # update colors of scatterplot
            if plot_dict["type"] == "scatter":
                collection_subplot = plot_dict["ax"].collections[0]
                collection_subplot.set_facecolors(self.plot_dicts[0]["fc"])

            # update colors of histograms
            elif plot_dict["type"] == "hist":
                if plot_dict["subtype"] == "1d_tour":
                    feature_selection = plot_dict["feature_selection"]
                    plot_dict["proj"][feature_selection, 0] = plot_dict["proj"][feature_selection, 0] / \
                        np.linalg.norm(
                            plot_dict["proj"][feature_selection, 0])
                    x = np.matmul(plot_dict["data"][:, feature_selection],
                                  plot_dict["proj"][feature_selection])/plot_dict["half_range"]
                    x = x[:, 0]
                else:
                    x = plot_dict["data"][:, plot_dict["hist_feature"]]

                # Get x and y_lims of old plot
                x_lims = plot_dict["ax"].get_xlim()
                y_lims = plot_dict["ax"].get_ylim()
                title = plot_dict["ax"].get_title()
                x_label = plot_dict["ax"].get_xlabel()

                x_subselections = []
                for subselection in self.plot_dicts[0]["subselections"]:
                    if subselection.shape[0] != 0:
                        x_subselections.append(x[subselection])
                    else:
                        x_subselections.append(np.array([]))

                plot_dict["ax"].clear()
                plot_dict["ax"].hist(
                    x_subselections,
                    stacked=True,
                    picker=True,
                    color=self.colors[:len(x_subselections)])

                plot_dict["ax"].set_ylim(y_lims)
                plot_dict["ax"].set_xlim(x_lims)
                plot_dict["ax"].set_title(title)
                plot_dict["ax"].set_xlabel(x_label)

        self.canvas.draw_idle()

    def disconnect(self):
        self.canvas.mpl_disconnect(self.connection)
        self.canvas.draw_idle()


class DraggableAnnotation1d:
    def __init__(self, data, plot_dicts, subplot_idx, hist, half_range, feature_selection, colors, labels):
        self.data = data
        self.plot_dicts = plot_dicts
        self.subplot_idx = subplot_idx
        self.feature_selection = feature_selection
        self.colors = colors
        self.proj = plot_dicts[subplot_idx]["proj"]
        self.proj.setflags(write=True)
        self.press = None
        self.ax = plot_dicts[subplot_idx]["ax"]
        self.hist = hist
        self.half_range = half_range

        self.alpha_other = 0.3
        self.arrs = []
        self.labels = []

        # Receive full projection
        self.proj[self.feature_selection, 0] = self.proj[self.feature_selection, 0] / \
            np.linalg.norm(self.proj[self.feature_selection, 0])

        divider = make_axes_locatable(self.ax)
        self.arrow_axs = divider.append_axes(
            "bottom", 1, pad=0.1)
        self.ax.tick_params(axis="x", labelbottom=False)
        self.arrow_axs.tick_params(
            axis="y", which="both", left=False, labelleft=False)
        self.arrow_axs.set_ylim(-0.05, 1.05)
        self.arrow_axs.set_xlim(-1, 1)

        for axis_id, feature_bool in enumerate(self.feature_selection):
            if feature_bool == True:
                arr = self.arrow_axs.arrow(0, axis_id/len(self.feature_selection),
                                           self.proj[axis_id, 0], 0,
                                           head_width=0.1,
                                           length_includes_head=True)

                label = self.arrow_axs.text(self.proj[axis_id],
                                            axis_id /
                                            len(self.feature_selection),
                                            labels[axis_id])

                self.cidpress = arr.figure.canvas.mpl_connect(
                    "button_press_event", self.on_press)
                self.cidrelease = arr.figure.canvas.mpl_connect(
                    "button_release_event", self.on_release)
                self.cidmotion = arr.figure.canvas.mpl_connect(
                    "motion_notify_event", self.on_motion)
            else:
                arr = None
                label = self.arrow_axs.text(0,
                                            axis_id /
                                            len(self.feature_selection),
                                            labels[axis_id])
            self.arrs.append(arr)
            self.labels.append(label)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        # Iterate through projection axes
        for axis_id, arr in enumerate(self.arrs):
            if arr is not None:
                if event.inaxes == arr.axes and event.button == 3:
                    contains, attrd = arr.contains(event)
                    if contains:
                        self.press = axis_id

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if self.press is None:
            return
        axis_id = self.press
        if event.xdata and event.ydata is not False:
            # Update projections
            self.proj[axis_id] = event.xdata
            # Orthonormalize
            self.proj[self.feature_selection, 0] = self.proj[self.feature_selection, 0] / \
                np.linalg.norm(self.proj[self.feature_selection, 0])

            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    self.arrs[axis_id].remove()
                    self.arrs[axis_id] = self.arrow_axs.arrow(0, axis_id/len(self.feature_selection),
                                                              self.proj[axis_id, 0], 0,
                                                              head_width=0.1,
                                                              length_includes_head=True)
                    # Update labels
                    self.labels[axis_id].set_x(self.proj[axis_id])

            # Update scattplot locations
            x = np.matmul(self.data[:, self.feature_selection],
                          self.proj[self.feature_selection])/self.half_range
            x = x[:, 0]
            self.plot_dicts[self.subplot_idx]["x"] = x
            # self.scat.set_offsets(new_data)
            title = self.ax.get_title()
            x_label = self.ax.get_xlabel()

            self.ax.clear()
            # check if there are preselected points and update plot
            x_subselections = []
            for subselection in self.plot_dicts[0]["subselections"]:
                if subselection.shape[0] != 0:
                    x_subselections.append(x[subselection])
                else:
                    x_subselections.append(np.array([]))
            self.plot_dicts[self.subplot_idx]["ax"].clear()
            self.plot_dicts[self.subplot_idx]["ax"].hist(
                x_subselections,
                stacked=True,
                picker=True,
                color=self.colors[:len(x_subselections)])

            bar_selector = BarSelect(plot_dicts=self.plot_dicts,
                                     subplot_idx=self.subplot_idx,
                                     feature_selection=self.feature_selection,
                                     colors=self.colors,
                                     half_range=self.half_range,
                                     alpha_other=self.alpha_other)
            self.plot_dicts[self.subplot_idx]["selector"] = bar_selector

            # redraw
            self.ax.tick_params(axis="x", labelbottom=False)
            self.arrow_axs.tick_params(axis="y", which="both", labelleft=False)
            self.ax.set_title(title)
            self.ax.set_xlabel(x_label)
            self.ax.set_xlim(-1, 1)
            self.ax.figure.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        self.press = None

    def remove(self):
        self.arrow_axs.remove()


class DraggableAnnotation2d:
    def __init__(self, data, proj, ax, scat, half_range, feature_selection, labels):
        self.data = data
        self.feature_selection = feature_selection
        self.proj = proj
        self.proj.setflags(write=True)
        self.press = None
        self.ax = ax
        self.scat = scat
        self.half_range = half_range

        self.arrs = []
        self.labels = []
        # Receive full projection
        self.proj[self.feature_selection, 0] = self.proj[self.feature_selection, 0] / \
            np.linalg.norm(self.proj[self.feature_selection, 0])
        self.proj[self.feature_selection, 1] = gram_schmidt(
            self.proj[self.feature_selection, 0], self.proj[self.feature_selection, 1])
        self.proj[self.feature_selection, 1] = self.proj[self.feature_selection, 1] / \
            np.linalg.norm(self.proj[self.feature_selection, 1])

        for axis_id, feature_bool in enumerate(self.feature_selection):
            if feature_bool == True:
                arr = self.ax.arrow(0, 0,
                                    self.proj[axis_id, 0]*2/3,
                                    self.proj[axis_id, 1]*2/3,
                                    head_width=0.06,
                                    length_includes_head=True)

                label = self.ax.text(self.proj[axis_id, 0]*2/3,
                                     self.proj[axis_id, 1]*2/3,
                                     labels[axis_id])

                self.cidpress = arr.figure.canvas.mpl_connect(
                    "button_press_event", self.on_press)
                self.cidrelease = arr.figure.canvas.mpl_connect(
                    "button_release_event", self.on_release)
                self.cidmotion = arr.figure.canvas.mpl_connect(
                    "motion_notify_event", self.on_motion)
            else:
                arr = None
                label = None
            self.arrs.append(arr)
            self.labels.append(label)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        # Iterate through projection axes
        for axis_id, arr in enumerate(self.arrs):
            if arr is not None:
                if event.inaxes == arr.axes and event.button == 3:
                    contains, attrd = arr.contains(event)
                    if contains:
                        self.press = axis_id

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if self.press is None:
            return
        axis_id = self.press
        if event.xdata and event.ydata is not False:
            # Update projections
            self.proj[axis_id] = [event.xdata/(2/3), event.ydata/(2/3)]
            # Orthonormalize
            self.proj[self.feature_selection, 0] = self.proj[self.feature_selection, 0] / \
                np.linalg.norm(self.proj[self.feature_selection, 0])
            self.proj[self.feature_selection, 1] = gram_schmidt(
                self.proj[self.feature_selection, 0], self.proj[self.feature_selection, 1])
            self.proj[self.feature_selection, 1] = self.proj[self.feature_selection, 1] / \
                np.linalg.norm(self.proj[self.feature_selection, 1])

            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    self.arrs[axis_id].remove()
                    self.arrs[axis_id] = self.ax.arrow(0, 0,
                                                       self.proj[axis_id,
                                                                 0]*2/3,
                                                       self.proj[axis_id,
                                                                 1]*2/3,
                                                       head_width=0.06,
                                                       length_includes_head=True)

                    # Update labels
                    self.labels[axis_id].set_x(self.proj[axis_id, 0]*2/3)
                    self.labels[axis_id].set_y(self.proj[axis_id, 1]*2/3)

            # Update scattplot locations
            new_data = np.matmul(self.data[:, self.feature_selection],
                                 self.proj[self.feature_selection])/self.half_range
            self.scat.set_offsets(new_data)

            # redraw
            self.ax.figure.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        self.press = None

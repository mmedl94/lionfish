import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from functools import partial

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, SpanSelector
from matplotlib.patches import Rectangle

from helpers import gram_schmidt
# Helper class that manages the lasso selection


class LassoSelect:
    def __init__(self, plot_dicts, subplot_idx, alpha_other=0.3, last_selection=False):
        # initialize arguments
        self.plot_dicts = plot_dicts
        self.canvas = plot_dicts[subplot_idx]["ax"].figure.canvas
        self.collection = plot_dicts[subplot_idx]["ax"].collections[0]
        self.alpha_other = alpha_other
        self.last_selection = last_selection

        # Get color of data points in RGB and construct data frame describing coloration
        self.fc = self.collection.get_facecolors()

        plot_dicts[subplot_idx]["ax"].collections[0]
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (len(self.collection.get_offsets()), 1))

        # initialize lasso selector
        self.lasso = LassoSelector(
            plot_dicts[subplot_idx]["ax"],
            onselect=partial(self.onselect, last_selection),
            button=1)
        self.ind = []

    # onselect governs what happens with selected data points
    # changes alpha of selected data points
    # saves indices of selected data points
    def onselect(self, last_selection, verts):
        path = Path(verts)
        xys = self.collection.get_offsets()
        self.ind = np.nonzero(path.contains_points(xys))[0]
        last_selection[0] = list(self.ind)
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        # update other plots if applicable
        for plot_dict in self.plot_dicts:
            # check plots if they are scatterplots. if so recolor datapoints
            if plot_dict["type"] == "scatter":
                collection_subplot = plot_dict["ax"].collections[0]
                collection_subplot.set_facecolors(self.fc)
            elif plot_dict["type"] == "hist":
                # Unpack histogram
                selected_obs = plot_dict["data"][self.ind]
                other_obs = np.delete(plot_dict["data"], self.ind)

                fc_sel = plot_dict["fc"]
                fc_not_sel = fc_sel.copy()
                fc_not_sel[-1] = self.alpha_other
                color_map = [plot_dict["fc"], fc_not_sel]
                # Get x and y_lims of old plot
                x_lims = plot_dict["ax"].get_xlim()
                y_lims = plot_dict["ax"].get_ylim()
                title = plot_dict["ax"].get_title()
                x_label = plot_dict["ax"].get_xlabel()

                plot_dict["ax"].clear()
                plot_dict["ax"].hist(
                    [selected_obs, other_obs],
                    stacked=True,
                    picker=True,
                    color=color_map)
                if plot_dict["vlines"] is not False:
                    plot_dict["vlines"].remove()

                plot_dict["ax"].set_ylim(y_lims)
                plot_dict["ax"].set_xlim(x_lims)
                plot_dict["ax"].set_title(title)
                plot_dict["ax"].set_xlabel(x_label)

                if selected_obs.shape[0] != 0:
                    vlines = plot_dict["ax"].vlines([selected_obs.min(), selected_obs.max()],
                                                    y_lims[0],
                                                    y_lims[1], color="red")
                    plot_dict["vlines"] = vlines
                else:
                    plot_dict["vlines"] = False

        self.canvas.draw_idle()

    # governs what happens when disconnected (after pressing "enter")
    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


class BarSelect:
    def __init__(self, plot_dicts, subplot_idx, alpha_other=0.3, last_selection=False):
        # initialize parameters
        self.plot_dicts = plot_dicts
        self.subplot_idx = subplot_idx
        self.ax = plot_dicts[subplot_idx]["ax"]
        self.data = plot_dicts[subplot_idx]["data"]
        self.canvas = self.ax.figure.canvas
        self.collection = self.ax.collections
        self.alpha_other = alpha_other
        self.last_selection = last_selection
        self.patches = self.ax.patches
        self.old_fc = list(self.ax.patches[0].get_facecolor())
        self.old_fc[-1] = 1
        self.new_fc = self.old_fc.copy()
        self.new_fc[-1] = self.alpha_other
        self.y_lims = self.ax.get_ylim()

        if plot_dicts[subplot_idx]["vlines"] is not False:
            self.ax.set_ylim(self.y_lims)
            if isinstance(plot_dicts[subplot_idx]["vlines"], list):
                vline_x = plot_dicts[subplot_idx]["vlines"]
            else:
                segments = plot_dicts[subplot_idx]["vlines"].get_segments()
                vline_x = np.unique([segment[0][0] for segment in segments])
            plot_dicts[subplot_idx]["vlines"] = self.ax.vlines(vline_x,
                                                               self.y_lims[0],
                                                               self.y_lims[1], color="red")

        self.connection = self.ax.figure.canvas.mpl_connect('pick_event', partial(
            self.onselect, self.last_selection))
        self.ind = []

    # onselect governs what happens with selected data points
    # changes alpha of selected data points
    # saves indices of selected data points
    def onselect(self, last_selection, event):
        if event.artist.axes != self.ax:
            return

        min_select = event.artist.get_x()
        max_select = min_select+event.artist.get_width()

        # Handle selection behaviour
        print(last_selection[0])
        if not last_selection[0]:
            # save selection as shared last_selection object
            last_selection[0] = np.where(np.logical_and(
                self.data >= min_select, self.data <= max_select))[0].tolist()
        else:
            new_ind = np.where(np.logical_and(
                self.data >= min_select, self.data <= max_select))[0].tolist()
            if set(list(new_ind)).issubset(set(last_selection[0])):
                last_selection[0] = list(
                    set(last_selection[0])-set(list(new_ind)))
            else:
                last_selection[0] = last_selection[0] + list(new_ind)
                last_selection[0] = list(set(last_selection[0]))

        for plot_dict in self.plot_dicts:
            # update colors of scatterplot
            if plot_dict["type"] == "scatter":
                scatter_fc = np.tile(self.old_fc, (self.data.shape[0], 1))
                scatter_fc[:, -1] = self.alpha_other
                scatter_fc[last_selection[0], -1] = 1
                collection_subplot = plot_dict["ax"].collections[0]
                collection_subplot.set_facecolors(scatter_fc)
            # update colors and vlines of histograms
            elif plot_dict["type"] == "hist":
                # Unpack histogram
                selected_obs = plot_dict["data"][last_selection[0]]
                other_obs = np.delete(plot_dict["data"], last_selection[0])

                fc_sel = plot_dict["fc"]
                fc_not_sel = fc_sel.copy()
                fc_not_sel[-1] = self.alpha_other
                color_map = [plot_dict["fc"], fc_not_sel]
                # Get x and y_lims of old plot
                x_lims = plot_dict["ax"].get_xlim()
                y_lims = plot_dict["ax"].get_ylim()
                title = plot_dict["ax"].get_title()
                x_label = plot_dict["ax"].get_xlabel()

                plot_dict["ax"].clear()
                plot_dict["ax"].hist(
                    [selected_obs, other_obs],
                    stacked=True,
                    picker=True,
                    color=color_map)
                if plot_dict["vlines"] is not False:
                    plot_dict["vlines"].remove()

                plot_dict["ax"].set_ylim(y_lims)
                plot_dict["ax"].set_xlim(x_lims)
                plot_dict["ax"].set_title(title)
                plot_dict["ax"].set_xlabel(x_label)

                # Redraw vlines if there are selected points
                if selected_obs.shape[0] != 0:
                    vlines = plot_dict["ax"].vlines([selected_obs.min(), selected_obs.max()],
                                                    y_lims[0],
                                                    y_lims[1], color="red")
                    plot_dict["vlines"] = vlines
                else:
                    plot_dict["vlines"] = False
        self.canvas.draw_idle()

    def disconnect(self):
        self.canvas.mpl_disconnect(self.connection)
        self.canvas.draw_idle()


class DraggableAnnotation:
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

        for axis_id, feature_bool in enumerate(self.feature_selection):
            if feature_bool == True:
                arr = self.ax.arrow(0, 0,
                                    proj[axis_id, 0],
                                    proj[axis_id, 1],
                                    head_width=0.06,
                                    length_includes_head=True)

                label = self.ax.text(proj[axis_id, 0],
                                     proj[axis_id, 1],
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
            self.proj[axis_id] = [event.xdata, event.ydata]

            # Orthonormalize
            self.proj[:, 0] = self.proj[:, 0]/np.linalg.norm(self.proj[:, 0])
            self.proj[:, 1] = gram_schmidt(self.proj[:, 0], self.proj[:, 1])
            self.proj[:, 1] = self.proj[:, 1]/np.linalg.norm(self.proj[:, 1])

            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    self.arrs[axis_id].remove()
                    self.arrs[axis_id] = self.ax.arrow(0, 0,
                                                       self.proj[axis_id, 0],
                                                       self.proj[axis_id, 1],
                                                       head_width=0.06,
                                                       length_includes_head=True)

                    # Update labels
                    self.labels[axis_id].set_x(self.proj[axis_id, 0])
                    self.labels[axis_id].set_y(self.proj[axis_id, 1])

            # Update scattplot locations
            new_data = np.matmul(self.data[:, self.feature_selection],
                                 self.proj[self.feature_selection])/self.half_range
            self.scat.set_offsets(new_data)

            # redraw
            self.ax.figure.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        self.press = None

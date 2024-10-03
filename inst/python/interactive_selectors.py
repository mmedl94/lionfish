import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from functools import partial
from itertools import product
import time

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.graphics.mosaicplot import mosaic

from helpers import gram_schmidt
# Helper class that manages the lasso selection


class LassoSelect:
    def __init__(self, parent, subplot_idx):
        # initialize arguments
        self.parent = parent
        self.n_pts = parent.n_pts
        self.plot_dicts = parent.plot_dicts
        self.subplot_idx = subplot_idx
        self.ax = parent.axs[subplot_idx]
        self.canvas = self.ax.figure.canvas
        self.collection = self.ax.collections[0]
        self.colors = parent.colors
        self.data = parent.data

    def get_blit(self):
        self.blit = self.ax.figure.canvas.copy_from_bbox(
            self.ax.bbox)
        # initialize lasso selector
        self.lasso = LassoSelector(
            self.ax,
            onselect=partial(self.onselect),
            button=1,
            useblit=True)
        self.lasso.background = self.blit
        self.ind = []

    # onselect governs what happens with selected data points
    # changes alpha of selected data points
    # saves indices of selected data points

    def onselect(self, verts):
        path = Path(verts)
        xys = self.collection.get_offsets()
        self.ind = np.nonzero(path.contains_points(xys))[0]

        # Check which subset is active
        for col_idx, subselection_var in enumerate(self.parent.subselection_vars):
            # If subset is active
            if subselection_var.get() == 1:
                # change facecolors
                self.parent.fc[self.ind] = [
                    self.colors[col_idx]]
                # Change subselections
                for idx, subselection in enumerate(self.parent.subselections):
                    # if the looped over subset isn't the currently selected, remove
                    # newly selected indices from old subsets
                    if col_idx != idx:
                        if set(self.ind) & set(subselection):
                            updated_ind = np.setdiff1d(subselection, self.ind)
                            self.parent.subselections[idx] = updated_ind

                # get set of old selection and new selection
                selected_set = list(set(self.ind).union(
                    set(self.parent.subselections[col_idx])))
                self.parent.subselections[col_idx] = np.array(
                    selected_set)

        self.collection.set_facecolors(self.parent.fc)
        for plot_idx, _ in enumerate(self.plot_dicts):
            self.plot_dicts[plot_idx]["update_plot"] = False
        self.parent.frame_update = False

        # Start new plot loop
        self.parent.after(10, self.parent.plot_loop)

    def disconnect(self):
        self.lasso.disconnect_events()


class BarSelect:
    def __init__(self, parent, subplot_idx):
        # initialize parameters
        self.parent = parent
        self.plot_dicts = parent.plot_dicts
        self.subplot_idx = subplot_idx
        self.plot_dict = parent.plot_dicts[subplot_idx]
        self.feature_selection = parent.feature_selection
        self.half_range = parent.half_range
        self.ax = parent.axs[subplot_idx]
        self.data = parent.data
        self.subtype = parent.plot_dicts[subplot_idx]["subtype"]
        if self.subtype == "hist":
            self.hist_feature = parent.plot_dicts[subplot_idx]["hist_feature"]

        self.canvas = self.ax.figure.canvas
        self.collection = self.ax.collections
        self.patches = self.ax.patches
        self.y_lims = self.ax.get_ylim()

        self.connection = self.ax.figure.canvas.mpl_connect("pick_event", partial(
            self.onselect))
        self.ind = []

    def get_blit(self):
        pass

    # onselect governs what happens with selected data points
    # changes alpha of selected data points
    # saves indices of selected data points

    def onselect(self, event):
        if event.artist.axes != self.ax:
            return

        min_select = event.artist.get_x()
        max_select = min_select+event.artist.get_width()

        cur_plot_dict = self.plot_dicts[self.subplot_idx]
        feature_selection = self.parent.feature_selection

        if cur_plot_dict["subtype"] == "1d_tour":
            cur_plot_dict["proj"][feature_selection, 0] = cur_plot_dict["proj"][feature_selection, 0] / \
                np.linalg.norm(
                    cur_plot_dict["proj"][feature_selection, 0])
            x = np.matmul(self.data[:, feature_selection],
                          cur_plot_dict["proj"][feature_selection])/self.half_range
            x = x[:, 0]
            x = x-np.mean(x)
            cur_plot_dict["x"] = x

        new_ind = np.where(np.logical_and(
            cur_plot_dict["x"] >= min_select,
            cur_plot_dict["x"] <= max_select))[0].tolist()

        # Check which subset is active
        for col_idx, subselection_var in enumerate(self.parent.subselection_vars):
            # If subset is active
            if subselection_var.get() == 1:
                # add new_ind to old selection
                merged_selection = list(
                    self.parent.subselections[col_idx]) + list(new_ind)
                merged_selection = np.array(list(set(merged_selection)))
                self.parent.subselections[col_idx] = merged_selection

                # remove new selection from other selections
                for idx, subselection in enumerate(self.parent.subselections):
                    # if the looped over subset isn't the currently selected, remove
                    # newly selected indices from old subsets
                    if col_idx != idx:
                        removed_selection = np.setdiff1d(
                            subselection, new_ind)
                        self.parent.subselections[idx] = removed_selection

        for col_idx, subselection in enumerate(self.parent.subselections):
            if subselection.shape[0] != 0:
                if self.parent.fc.shape[0] != 0:
                    color = self.parent.colors[col_idx].copy()
                    if color[-1] != 1:
                        color[-1] = 0.1
                    self.parent.fc[subselection] = color

        for plot_idx, _ in enumerate(self.plot_dicts):
            self.plot_dicts[plot_idx]["update_plot"] = False

        self.parent.frame_update = False
        self.parent.plot_loop()

    def disconnect(self):
        self.canvas.mpl_disconnect(self.connection)


class DraggableAnnotation1d:
    def __init__(self, parent, subplot_idx, hist):
        self.parent = parent
        self.data = parent.data
        self.plot_dicts = parent.plot_dicts
        self.subplot_idx = subplot_idx
        self.feature_selection = parent.feature_selection
        self.colors = parent.colors
        self.proj = self.plot_dicts[subplot_idx]["proj"]
        self.proj.setflags(write=True)
        self.press = None
        self.ax = parent.axs[subplot_idx]
        self.hist = hist
        self.half_range = parent.half_range

        self.arrs = []
        self.labels = []

        if sum(self.feature_selection) > self.parent.hover_cutoff:
            self.alpha = 0.1
        else:
            self.alpha = 1

        # Receive full projection
        self.proj[self.feature_selection, 0] = self.proj[self.feature_selection, 0] / \
            np.linalg.norm(self.proj[self.feature_selection, 0])

        divider = make_axes_locatable(self.ax)
        self.arrow_axs = divider.append_axes(
            "bottom", 1, pad=0.1)
        self.ax.tick_params(axis="x", labelbottom=False)
        self.arrow_axs.set_ylim(-0.05, 1.05)
        self.arrow_axs.set_xlim(-1, 1)
        self.arrow_axs.set_xticks([])
        self.arrow_axs.set_yticks([])

        true_counter = 0
        for axis_id, feature_bool in enumerate(self.feature_selection):
            if feature_bool == True:
                if len(self.feature_selection) < self.parent.hover_cutoff:
                    x_0 = 0
                    y_0 = axis_id/len(self.feature_selection)
                    dx = self.proj[axis_id, 0]
                    dy = 0
                else:
                    true_counter += 1
                    x_0 = 0
                    y_0 = true_counter/sum(self.feature_selection)
                    dx = self.proj[axis_id, 0]
                    dy = 0

                arr = self.arrow_axs.arrow(x_0, y_0,
                                           dx, dy,
                                           head_width=0.1,
                                           length_includes_head=True)

                label = self.arrow_axs.text(dx, y_0,
                                            parent.feature_names[axis_id],
                                            alpha=self.alpha,
                                            clip_on=True,
                                            size=self.parent.label_size)

            else:
                arr = None
                if len(self.feature_selection) < self.parent.hover_cutoff:
                    label = self.arrow_axs.text(0,
                                                axis_id /
                                                len(self.feature_selection),
                                                parent.feature_names[axis_id],
                                                alpha=self.alpha,
                                                size=self.parent.label_size)
                else:
                    label = None

            self.arrs.append(arr)
            self.labels.append(label)

        if self.parent.blendout_projection_switch.get():
            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    axes_blendout_threshold = float(
                        self.parent.blendout_projection_variable.get())
                    if (self.proj[axis_id] < axes_blendout_threshold) and \
                            (self.proj[axis_id] > -axes_blendout_threshold):
                        self.labels[axis_id].set_alpha(0)
                        self.arrs[axis_id].set_alpha(0)
                    else:
                        self.arrs[axis_id].set_alpha(1)

        self.cidpress = parent.fig.canvas.mpl_connect(
            "button_press_event", self.on_press)
        self.cidrelease = parent.fig.canvas.mpl_connect(
            "button_release_event", self.on_release)
        self.cidmotion = parent.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        # Iterate through projection axes
        for axis_id, arr in enumerate(self.arrs):
            if arr is not None:
                if event.inaxes == arr.axes and event.button == 3:
                    contains, attrd = arr.contains(event)
                    if contains:
                        self.press = axis_id

    def blend_out(self):
        self.collections = []
        self.patches = []
        self.texts = []

        for collection in self.ax.collections:
            self.collections.append(collection.get_alpha())
            collection.set_alpha(0)
        for patch in self.ax.patches:
            self.patches.append(patch.get_alpha())
            patch.set_alpha(0)
        for text in self.ax.texts:
            self.texts.append(text.get_alpha())
            text.set_alpha(0)

        self.arr_collections = []
        self.arr_patches = []
        self.arr_texts = []

        for collection in self.arrow_axs.collections:
            self.arr_collections.append(collection.get_alpha())
            collection.set_alpha(0)
        for patch in self.arrow_axs.patches:
            self.arr_patches.append(patch.get_alpha())
            patch.set_alpha(0)
        for text in self.arrow_axs.texts:
            self.arr_texts.append(text.get_alpha())
            text.set_alpha(0)

    def get_blit(self):
        bbox_mins = self.arrow_axs.bbox.get_points()[0]
        bbox_maxs = self.ax.bbox.get_points()[1]
        self.bbox = Bbox((bbox_mins, bbox_maxs))

        self.blit = self.ax.figure.canvas.copy_from_bbox(self.bbox)

    def blend_in(self):
        self.ax.figure.canvas.restore_region(self.blit)
        for idx, collection in enumerate(self.ax.collections):
            collection.set_alpha(self.collections[idx])
            self.ax.draw_artist(collection)
        for idx, patch in enumerate(self.ax.patches):
            patch.set_alpha(self.patches[idx])
            self.ax.draw_artist(patch)
        for idx, text in enumerate(self.ax.texts):
            text.set_alpha(self.texts[idx])
            self.ax.draw_artist(text)

        for idx, collection in enumerate(self.arrow_axs.collections):
            collection.set_alpha(self.arr_collections[idx])
            self.arrow_axs.draw_artist(collection)
        for idx, patch in enumerate(self.arrow_axs.patches):
            patch.set_alpha(self.arr_patches[idx])
            self.arrow_axs.draw_artist(patch)
        for idx, text in enumerate(self.arrow_axs.texts):
            text.set_alpha(self.arr_texts[idx])
            self.arrow_axs.draw_artist(text)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if event.inaxes != self.arrow_axs:
            return
        self.ax.figure.canvas.restore_region(self.blit)
        self.proj = self.plot_dicts[self.subplot_idx]["proj"]

        if self.press is not None:
            axis_id = self.press
            if event.xdata and event.ydata is not False:
                # Update projections
                self.proj[axis_id] = event.xdata
                # Orthonormalize
                self.proj[self.feature_selection, 0] = self.proj[self.feature_selection, 0] / \
                    np.linalg.norm(self.proj[self.feature_selection, 0])

                for axis_id, feature_bool in enumerate(self.feature_selection):
                    if feature_bool == True:
                        self.arrs[axis_id].set_data(dx=self.proj[axis_id, 0])
                        if self.labels[axis_id] != None:
                            # Update labels
                            self.labels[axis_id].set_x(self.proj[axis_id])

                x = np.matmul(self.data[:, self.feature_selection],
                              self.proj[self.feature_selection])/self.half_range
                x = x[:, 0]
                x = x - np.mean(x)

                self.plot_dicts[self.subplot_idx]["x"] = x
                self.plot_dicts[self.subplot_idx]["proj"] = self.proj

                # check if there are preselected points and update plot
                x_subselections = []
                for subselection in self.parent.subselections:
                    if subselection.shape[0] != 0:
                        x_subselections.append(x[subselection])
                    else:
                        x_subselections.append(np.array([]))
                xlim = self.ax.get_xlim()
                self.ax.clear()
                self.ax.hist(
                    x_subselections,
                    stacked=True,
                    picker=True,
                    color=self.colors[:len(x_subselections)],
                    animated=True,
                    bins=np.linspace(-1, 1, int(self.parent.n_bins.get())))

                self.ax.set_xlim(xlim)
                self.ax.set_xticks([])
                self.ax.set_yticks([])

                self.plot_dicts[self.subplot_idx]["selector"].disconnect()
                bar_selector = BarSelect(parent=self.parent,
                                         subplot_idx=self.subplot_idx)
                self.plot_dicts[self.subplot_idx]["selector"] = bar_selector

        if self.parent.blendout_projection_switch.get():
            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    axes_blendout_threshold = float(
                        self.parent.blendout_projection_variable.get())
                    if (self.proj[axis_id] < axes_blendout_threshold) and \
                            (self.proj[axis_id] > -axes_blendout_threshold):
                        self.labels[axis_id].set_alpha(0)
                        self.arrs[axis_id].set_alpha(0)
                    else:
                        self.arrs[axis_id].set_alpha(1)
                        self.labels[axis_id].set_alpha(1)

        if self.alpha != 1:
            for label_idx, label in enumerate(self.labels):
                if label:
                    label_pos = label.get_position()
                    if (label_pos[0] > event.xdata-0.1) and (label_pos[0] < event.xdata+0.1) and \
                            (label_pos[1] > event.ydata-0.1) and (label_pos[1] < event.ydata+0.1):
                        if self.labels[label_idx].get_alpha() != 0:
                            self.labels[label_idx].set_alpha(1)
                    else:
                        if self.labels[label_idx].get_alpha() != 0:
                            self.labels[label_idx].set_alpha(0.1)

        for collection in self.ax.collections:
            self.ax.draw_artist(collection)
        for patch in self.ax.patches:
            self.ax.draw_artist(patch)
        for text in self.ax.texts:
            self.ax.draw_artist(text)

        for collection in self.arrow_axs.collections:
            self.arrow_axs.draw_artist(collection)
        for patch in self.arrow_axs.patches:
            self.arrow_axs.draw_artist(patch)
        for text in self.arrow_axs.texts:
            self.arrow_axs.draw_artist(text)

        self.ax.figure.canvas.blit(self.bbox)

    def on_release(self, event):
        """Clear button press information."""
        self.press = None

    def update(self, plot_object, frame):
        self.ax.figure.canvas.restore_region(self.blit)

        frame = int(self.parent.frame_vars[self.subplot_idx].get())

        if frame >= plot_object["obj"].shape[-1]-1:
            frame = plot_object["obj"].shape[-1]-1
            self.parent.frame_vars[self.subplot_idx].set(str(frame))

        self.proj = np.copy(plot_object["obj"][:, :, frame])
        self.proj[self.feature_selection, 0] = self.proj[self.feature_selection, 0] / \
            np.linalg.norm(self.proj[self.feature_selection, 0])

        for axis_id, feature_bool in enumerate(self.parent.feature_selection):
            if feature_bool == True:
                self.arrs[axis_id].set_data(dx=self.proj[axis_id, 0])
                if self.labels[axis_id] != None:
                    # Update labels
                    self.labels[axis_id].set_x(self.proj[axis_id])

        x = np.matmul(self.data[:, self.feature_selection],
                      self.proj[self.feature_selection])/self.half_range
        x = x[:, 0]
        x = x-np.mean(x)

        self.plot_dicts[self.subplot_idx]["x"] = x
        self.plot_dicts[self.subplot_idx]["proj"] = self.proj

        x_subselections = []
        for subselection in self.parent.subselections:
            if subselection.shape[0] != 0:
                x_subselections.append(x[subselection])
            else:
                x_subselections.append(np.array([]))

        xlim = self.ax.get_xlim()
        self.ax.clear()
        self.ax.hist(
            x_subselections,
            stacked=True,
            picker=True,
            color=self.colors[:len(x_subselections)],
            animated=True,
            bins=np.linspace(-1, 1, int(self.parent.n_bins.get())))

        self.ax.set_xlim(xlim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.plot_dicts[self.subplot_idx]["selector"].disconnect()
        bar_selector = BarSelect(parent=self.parent,
                                 subplot_idx=self.subplot_idx)
        self.plot_dicts[self.subplot_idx]["selector"] = bar_selector

        for collection in self.ax.collections:
            self.ax.draw_artist(collection)
        for patch in self.ax.patches:
            self.ax.draw_artist(patch)
        for text in self.ax.texts:
            self.ax.draw_artist(text)

        for collection in self.arrow_axs.collections:
            self.arrow_axs.draw_artist(collection)
        for patch in self.arrow_axs.patches:
            self.arrow_axs.draw_artist(patch)
        for text in self.arrow_axs.texts:
            self.arrow_axs.draw_artist(text)

        self.ax.figure.canvas.blit(self.bbox)

    def remove(self):
        self.arrow_axs.remove()

    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)


class DraggableAnnotation2d:
    def __init__(self, parent, plot_idx):
        self.parent = parent
        self.data = parent.data
        self.feature_selection = parent.feature_selection
        self.proj = parent.plot_dicts[plot_idx]["proj"]
        self.proj.setflags(write=True)
        self.press = None
        self.ax = parent.axs[plot_idx]
        self.half_range = parent.half_range
        self.pressing = 0
        self.plot_dict = parent.plot_dicts[plot_idx]
        self.plot_dicts = parent.plot_dicts
        self.plot_idx = plot_idx
        self.blit = None

        if sum(self.feature_selection) > self.parent.hover_cutoff:
            self.alpha = 0.1
        else:
            self.alpha = 1

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
                                    length_includes_head=True,
                                    animated=True)

                label = self.ax.text(self.proj[axis_id, 0]*2/3,
                                     self.proj[axis_id, 1]*2/3,
                                     parent.feature_names[axis_id],
                                     alpha=self.alpha,
                                     animated=True,
                                     clip_on=True,
                                     size=self.parent.label_size)

                self.ax.draw_artist(arr)
                self.ax.draw_artist(label)

            else:
                arr = None
                label = None

            self.arrs.append(arr)
            self.labels.append(label)

        if self.parent.blendout_projection_switch.get():
            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    axes_blendout_threshold = float(
                        self.parent.blendout_projection_variable.get())
                    proj_length = np.linalg.norm(self.proj[axis_id])
                    if (proj_length < axes_blendout_threshold) and \
                            (proj_length > -axes_blendout_threshold):
                        self.labels[axis_id].set_alpha(0)
                        self.arrs[axis_id].set_alpha(0)
                    else:
                        self.arrs[axis_id].set_alpha(1)

        self.connect()

    def blend_out(self):
        self.collections = []
        self.patches = []
        self.texts = []
        if self.ax.collections:
            for collection in self.ax.collections:
                self.collections.append(collection.get_alpha())
                collection.set_alpha(0)
        if self.ax.patches:
            for patch in self.ax.patches:
                self.patches.append(patch.get_alpha())
                patch.set_alpha(0)
        if self.ax.texts:
            for text in self.ax.texts:
                self.texts.append(text.get_alpha())
                text.set_alpha(0)

    def get_blit(self):
        self.blit = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)

    def blend_in(self):
        if self.ax.collections:
            for idx, collection in enumerate(self.ax.collections):
                collection.set_alpha(self.collections[idx])
                self.ax.draw_artist(collection)
        if self.ax.patches:
            for idx, patch in enumerate(self.ax.patches):
                patch.set_alpha(self.patches[idx])
                self.ax.draw_artist(patch)
        if self.ax.texts:
            for idx, text in enumerate(self.ax.texts):
                text.set_alpha(self.texts[idx])
                self.ax.draw_artist(text)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        # Iterate through projection axes
        self.pressing = 1
        for axis_id, arr in enumerate(self.arrs):
            if arr is not None:
                if event.inaxes == arr.axes and event.button == 3:
                    contains, attrd = arr.contains(event)
                    if contains:
                        self.press = axis_id

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if event.button == 1:
            return
        if event.inaxes != self.ax:
            return

        self.ax.figure.canvas.restore_region(self.blit)

        axis_id = self.press

        if self.press is not None:
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
                for proj_axis_id, feature_bool in enumerate(self.feature_selection):
                    if feature_bool == True:
                        self.arrs[proj_axis_id].set_data(x=0,
                                                         y=0,
                                                         dx=self.proj[proj_axis_id,
                                                                      0]*2/3,
                                                         dy=self.proj[proj_axis_id,
                                                                      1]*2/3)
                        # Update labels
                        self.labels[proj_axis_id].set_x(
                            self.proj[proj_axis_id, 0]*2/3)
                        self.labels[proj_axis_id].set_y(
                            self.proj[proj_axis_id, 1]*2/3)
                # Update scattplot locations
                new_data = np.matmul(self.data[:, self.feature_selection],
                                     self.proj[self.feature_selection])/self.half_range
                new_data = new_data - np.mean(new_data, axis=0)
                self.ax.collections[0].set_offsets(new_data)

        self.parent.plot_dicts[self.plot_idx]["proj"] = self.proj

        if self.parent.blendout_projection_switch.get():
            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    axes_blendout_threshold = float(
                        self.parent.blendout_projection_variable.get())
                    proj_length = np.linalg.norm(self.proj[axis_id])
                    if (proj_length < axes_blendout_threshold) and \
                            (proj_length > -axes_blendout_threshold):
                        self.labels[axis_id].set_alpha(0)
                        self.arrs[axis_id].set_alpha(0)
                    else:
                        self.labels[axis_id].set_alpha(1)
                        self.arrs[axis_id].set_alpha(1)

        if self.alpha != 1:
            for label_idx, label in enumerate(self.labels):
                if label:
                    label_pos = label.get_position()
                    if (label_pos[0] > event.xdata-0.1) and (label_pos[0] < event.xdata+0.1) and \
                            (label_pos[1] > event.ydata-0.1) and (label_pos[1] < event.ydata+0.1):
                        if self.labels[label_idx].get_alpha() != 0:
                            self.labels[label_idx].set_alpha(1)
                    else:
                        if self.labels[label_idx].get_alpha() != 0:
                            self.labels[label_idx].set_alpha(0.1)

        for collection in self.ax.collections:
            self.ax.draw_artist(collection)
        for patch in self.ax.patches:
            self.ax.draw_artist(patch)
        for text in self.ax.texts:
            self.ax.draw_artist(text)
        self.ax.figure.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        """Clear button press information."""
        self.pressing = 0
        self.press = None

    def blendout_update(self):
        self.ax.figure.canvas.restore_region(self.blit)

        if self.parent.blendout_projection_switch.get():
            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    axes_blendout_threshold = float(
                        self.parent.blendout_projection_variable.get())
                    proj_length = np.linalg.norm(self.proj[axis_id])
                    if (proj_length < axes_blendout_threshold) and \
                            (proj_length > -axes_blendout_threshold):
                        self.labels[axis_id].set_alpha(0)
                        self.arrs[axis_id].set_alpha(0)
                    else:
                        self.arrs[axis_id].set_alpha(1)
        else:
            for axis_id, feature_bool in enumerate(self.feature_selection):
                if feature_bool == True:
                    self.arrs[axis_id].set_alpha(1)
                    self.labels[axis_id].set_alpha(self.alpha)

        for collection in self.ax.collections:
            self.ax.draw_artist(collection)
        for patch in self.ax.patches:
            self.ax.draw_artist(patch)
        for text in self.ax.texts:
            self.ax.draw_artist(text)
        self.ax.figure.canvas.blit(self.ax.bbox)

    def update(self, plot_object, frame):
        self.ax.figure.canvas.restore_region(self.blit)

        frame = int(self.parent.frame_vars[self.plot_idx].get())

        if frame >= plot_object["obj"].shape[-1]-1:
            frame = plot_object["obj"].shape[-1]-1
            self.parent.frame_vars[self.plot_idx].set(str(frame))

        self.proj = np.copy(plot_object["obj"][:, :, frame])

        # Orthonormalize
        self.proj[self.feature_selection, 0] = self.proj[self.feature_selection, 0] / \
            np.linalg.norm(self.proj[self.feature_selection, 0])
        self.proj[self.feature_selection, 1] = gram_schmidt(
            self.proj[self.feature_selection, 0], self.proj[self.feature_selection, 1])
        self.proj[self.feature_selection, 1] = self.proj[self.feature_selection, 1] / \
            np.linalg.norm(self.proj[self.feature_selection, 1])

        for proj_axis_id, feature_bool in enumerate(self.feature_selection):
            if feature_bool == True:
                self.arrs[proj_axis_id].set_data(x=0,
                                                 y=0,
                                                 dx=self.proj[proj_axis_id,
                                                              0]*2/3,
                                                 dy=self.proj[proj_axis_id,
                                                              1]*2/3)
                # Update labels
                self.labels[proj_axis_id].set_x(
                    self.proj[proj_axis_id, 0]*2/3)
                self.labels[proj_axis_id].set_y(
                    self.proj[proj_axis_id, 1]*2/3)

        # Update scattplot locations
        new_data = np.matmul(self.data[:, self.feature_selection],
                             self.proj[self.feature_selection])/self.half_range
        new_data = new_data - np.mean(new_data, axis=0)
        self.ax.collections[0].set_offsets(new_data)

        self.parent.plot_dicts[self.plot_idx]["proj"] = self.proj

        for collection in self.ax.collections:
            self.ax.draw_artist(collection)
        for patch in self.ax.patches:
            self.ax.draw_artist(patch)
        for text in self.ax.texts:
            self.ax.draw_artist(text)

        self.ax.figure.canvas.blit(self.ax.bbox)

    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)

    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect(
            "button_release_event", self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion)

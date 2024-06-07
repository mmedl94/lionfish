import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from functools import partial

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, SpanSelector

# Helper class that manages the lasso selection


class LassoSelect:
    def __init__(self, plot_dicts, subplot_idx, alpha_other=0.3, last_selection=False):
        # initialize arguments
        self.plot_dicts = plot_dicts
        self.canvas = plot_dicts[subplot_idx]["ax"].figure.canvas
        self.collection = plot_dicts[subplot_idx]["ax"].collections[0]
        self.alpha_other = alpha_other
        self.last_selection = last_selection

        # Get coordinates and number of data points
        self.xys = self.collection.get_offsets()

        # Get color of data points in RGB and construct data frame describing coloration
        self.fc = self.collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (len(self.xys), 1))

        # initialize lasso selector
        self.lasso = LassoSelector(
            plot_dicts[subplot_idx]["ax"],
            onselect=partial(self.onselect, last_selection))
        self.ind = []

    # onselect governs what happens with selected data points
    # changes alpha of selected data points
    # saves indices of selected data points
    def onselect(self, last_selection, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        last_selection[0] = self.ind
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
                bins = plot_dict["bins"]
                patches = plot_dict["ax"].patches
                selected_obs = plot_dict["data"][self.ind]
                old_fc = list(patches[0].get_facecolor())
                old_fc[-1] = 1
                new_fc = old_fc.copy()
                new_fc[-1] = self.alpha_other
                for i in range(len(bins) - 1):
                    if np.any((selected_obs >= bins[i]) & (selected_obs <= bins[i + 1])):
                        patches[i].set_facecolor(old_fc)
                    else:
                        patches[i].set_facecolor(new_fc)
                if plot_dict["vlines"] is not False:
                    plot_dict["vlines"].remove()
                y_lims = plot_dict["ax"].get_ylim()
                plot_dict["ax"].set_ylim(y_lims)

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


class SpanSelect:
    def __init__(self, plot_dicts, subplot_idx, alpha_other=0.3, last_selection=False):
        # initialize parameters
        self.plot_dicts = plot_dicts
        self.subplot_idx = subplot_idx
        self.ax = plot_dicts[subplot_idx]["ax"]
        self.bins = plot_dicts[subplot_idx]["bins"]
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

        # initialize lasso selector
        self.span = SpanSelector(
            self.ax,
            onselect=partial(self.onselect, last_selection=last_selection),
            direction="horizontal")
        self.ind = []

    # onselect governs what happens with selected data points
    # changes alpha of selected data points
    # saves indices of selected data points
    def onselect(self, min_select, max_select, last_selection):
        # save selection as shared last_selection object
        self.ind = np.where(np.logical_and(
            self.data >= min_select, self.data <= max_select))[0]
        last_selection[0] = self.ind
        # get the selected observations via indices
        selected_obs = self.data[self.ind]
        # recolor hist
        for i in range(len(self.bins) - 1):
            if np.any((selected_obs >= self.bins[i]) & (selected_obs <= self.bins[i + 1])):
                self.patches[i].set_facecolor(self.old_fc)
            else:
                self.patches[i].set_facecolor(self.new_fc)

        if self.plot_dicts[self.subplot_idx]["vlines"] is not False:
            self.plot_dicts[self.subplot_idx]["vlines"].remove()
            self.plot_dicts[self.subplot_idx]["vlines"] = False

        # add vlines
        if selected_obs.shape[0] != 0:
            self.ax.set_ylim(self.y_lims)
            vlines = self.ax.vlines([selected_obs.min(), selected_obs.max()],
                                    self.y_lims[0],
                                    self.y_lims[1], color="red")
            self.plot_dicts[self.subplot_idx]["vlines"] = vlines
            self.canvas.draw_idle()

        # define facecolors for scatterplots
        self.fc = np.tile(self.old_fc, (self.data.shape[0], 1))
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1

        for plot_dict in self.plot_dicts:
            # update colors of scatterplot
            if plot_dict["type"] == "scatter":
                collection_subplot = plot_dict["ax"].collections[0]
                collection_subplot.set_facecolors(self.fc)
            # update colors and vlines of histograms
            elif plot_dict["type"] == "hist":
                # Unpack histogram
                bins = plot_dict["bins"]
                patches = plot_dict["ax"].patches
                selected_obs = plot_dict["data"][self.ind]
                old_fc = list(patches[0].get_facecolor())
                old_fc[-1] = 1
                new_fc = old_fc.copy()
                new_fc[-1] = self.alpha_other
                for i in range(len(bins) - 1):
                    if np.any((selected_obs >= bins[i]) & (selected_obs <= bins[i + 1])):
                        patches[i].set_facecolor(old_fc)
                    else:
                        patches[i].set_facecolor(new_fc)

                # check if we have previous vlines
                if plot_dict["vlines"] is not False:
                    plot_dict["vlines"].remove()

                y_lims = plot_dict["ax"].get_ylim()
                plot_dict["ax"].set_ylim(y_lims)

                # Redraw vlines if there are selected points
                if selected_obs.shape[0] != 0:
                    vlines = plot_dict["ax"].vlines([selected_obs.min(), selected_obs.max()],
                                                    y_lims[0],
                                                    y_lims[1], color="red")
                    plot_dict["vlines"] = vlines
                else:
                    plot_dict["vlines"] = False

    # governs what happens when disconnected (after pressing "enter")
    def disconnect(self):
        self.span.disconnect_events()
        self.canvas.draw_idle()

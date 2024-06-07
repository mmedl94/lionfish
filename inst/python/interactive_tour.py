import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from pytour_selectors import LassoSelect, SpanSelect


class InteractiveTourInterface(tk.Frame):
    def __init__(self, parent, data, col_names, plot_objects, half_range, n_max_cols):
        tk.Frame.__init__(self, parent)

        if not isinstance(plot_objects, list):
            plot_objects = [plot_objects]

        # if len(plot_objects[0]) == 2:
        #    [plot_objects] = np.expand_dims(plot_objects[0], axis=2)

        if half_range is None:
            print("Using adaptive half_range")
        else:
            print(f"Using half_range of {half_range}")

        limits = 1
        alpha_other = 0.3
        n_pts = data.shape[0]
        # Initialize self.obs_idx with all obs
        self.obs_idx_ = np.arange(0, data.shape[0])
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
        canvas = FigureCanvasTkAgg(fig)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas)
        toolbar.update()
        canvas.get_tk_widget().pack()

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
        parent.protocol("WM_DELETE_WINDOW", cleanup)

        def accept(event):
            if event.key == "right" or event.key == "left":
                selector.disconnect()
                fig.canvas.draw()
                if event.key == "right":
                    self.frame += 1
                if event.key == "left" and self.frame > 0:
                    self.frame -= 1
                self.pause_var.set(1)

        plot_dicts = [i for i, _ in enumerate(plot_objects)]

        self.last_selection = [False]

        while self.frame < self.n_frames:
            selectors = []
            for subplot_idx, plot_object in enumerate(plot_objects):
                frame = self.frame

                if plot_object["type"] == "2d_tour":
                    if frame >= plot_object["obj"].shape[-1]-1:
                        frame = plot_object["obj"].shape[-1]-1
                    # get tour data
                    plot_data = r.render_proj_inter(
                        data, plot_object["obj"][:, :, frame], limits=limits, half_range=half_range)
                    # Unpack tour data
                    data_prj = plot_data["data_prj"]
                    axes_prj = plot_data["axes"]
                    circle_prj = plot_data["circle"]
                    x = data_prj.iloc[:, 0]
                    y = data_prj.iloc[:, 1]

                    # clear old scatterplot
                    axs[subplot_idx].clear()
                    # Make new scatterplot
                    scat = axs[subplot_idx].scatter(x, y)
                    scat = axs[subplot_idx].collections[0]
                    axs[subplot_idx].set_xlim(-limits*1.1, limits*1.1)
                    axs[subplot_idx].set_ylim(-limits*1.1, limits*1.1)
                    axs[subplot_idx].set_box_aspect(aspect=1)

                    # Recolor preselected points
                    if self.last_selection[0] is not False:
                        fc = scat.get_facecolors()
                        fc = np.tile(fc, (n_pts, 1))
                        fc[:, -1] = alpha_other
                        fc[self.last_selection, -1] = 1
                        scat.set_facecolors(fc)

                    plot_dict = {"type": "scatter",
                                 "subtype": "2d_tour",
                                 "ax": axs[subplot_idx]
                                 }
                    plot_dicts[subplot_idx] = plot_dict
                    # start Lasso selector
                    selector = LassoSelect(
                        plot_dicts=plot_dicts,
                        subplot_idx=subplot_idx,
                        alpha_other=alpha_other,
                        last_selection=self.last_selection)
                    selectors.append(selector)

                    # plot axes and circle
                    for arrow in range(axes_prj.shape[0]):
                        axs[subplot_idx].arrow(axes_prj.iloc[arrow, 0],
                                               axes_prj.iloc[arrow, 1],
                                               axes_prj.iloc[arrow, 2],
                                               axes_prj.iloc[arrow, 3])

                        axs[subplot_idx].text(axes_prj.iloc[arrow, 2],
                                              axes_prj.iloc[arrow, 3],
                                              col_names[arrow])

                        axs[subplot_idx].plot(circle_prj.iloc[:, 0],
                                              circle_prj.iloc[:, 1])
                    n_frames = plot_object["obj"].shape[-1]-1
                    axs[subplot_idx].set_title(f"Frame {frame} out of {n_frames}" +
                                               f"\nPress right key for next frame" +
                                               f"\nPress left key for last frame")

                if plot_object["type"] == "1d_tour":
                    if frame >= plot_object["obj"].shape[-1]-1:
                        frame = plot_object["obj"].shape[-1]-1

                    x = np.matmul(data, plot_object["obj"][:, 0, frame])
                    x = x/half_range
                    axs[subplot_idx].clear()

                    # check if there are preselected points and update plot
                    if self.last_selection[0] is not False:
                        # recolor preselected points
                        selected_obs = x[self.last_selection[0]]
                        other_obs = np.delete(x, self.last_selection[0])
                        fc_sel = plot_dicts[subplot_idx]["fc"]
                        fc_sel[-1] = 1
                        fc_not_sel = fc_sel.copy()
                        fc_not_sel[-1] = alpha_other
                        color_map = [fc_sel, fc_not_sel]
                        hist = axs[subplot_idx].hist(
                            [selected_obs, other_obs],
                            stacked=True,
                            color=color_map)
                        if selected_obs.shape[0] != 0:
                            vlines = [selected_obs.min(), selected_obs.max()]
                        else:
                            vlines = False
                    else:
                        hist = axs[subplot_idx].hist(x)
                        axs[subplot_idx].set_box_aspect(aspect=1)
                        vlines = False
                        fc_sel = list(hist[2][0].get_facecolor())

                    plot_dict = {"type": "hist",
                                 "subtype": "1d_tour",
                                 "ax": axs[subplot_idx],
                                 "data": x,
                                 "vlines": vlines,
                                 "fc": fc_sel}
                    plot_dicts[subplot_idx] = plot_dict
                    # start area selector
                    selector = SpanSelect(
                        plot_dicts=plot_dicts,
                        subplot_idx=subplot_idx,
                        alpha_other=alpha_other,
                        last_selection=self.last_selection)
                    selectors.append(selector)

                    n_frames = plot_object["obj"].shape[-1]-1
                    axs[subplot_idx].set_xlim(-1, 1)
                    axs[subplot_idx].set_title(f"Frame {frame} out of {n_frames}" +
                                               f"\nPress right key for next frame" +
                                               f"\nPress left key for last frame")

                if plot_object["type"] == "scatter":
                    # get data
                    col_index_x = col_names.index(plot_object["obj"][0])
                    col_index_y = col_names.index(plot_object["obj"][1])
                    x = data[:, col_index_x]
                    y = data[:, col_index_y]

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
                        fc[:, -1] = alpha_other
                        fc[self.last_selection, -1] = 1
                        scat.set_facecolors(fc)

                    plot_dict = {"type": "scatter",
                                 "subtype": "scatter",
                                 "ax": axs[subplot_idx]
                                 }
                    plot_dicts[subplot_idx] = plot_dict
                    # start Lasso selector
                    selector = LassoSelect(
                        plot_dicts=plot_dicts,
                        subplot_idx=subplot_idx,
                        alpha_other=alpha_other,
                        last_selection=self.last_selection)
                    selectors.append(selector)
                    x_name = plot_object["obj"][0]
                    y_name = plot_object["obj"][1]
                    axs[subplot_idx].set_xlabel(x_name)
                    axs[subplot_idx].set_ylabel(y_name)
                    axs[subplot_idx].set_title(
                        f"Scatterplot of variables {x_name} and {y_name}")

                elif plot_object["type"] == "hist":
                    if plot_object["obj"] in col_names:
                        col_index = col_names.index(plot_object["obj"])
                        x = data[:, col_index]
                        # clear old histogram
                        axs[subplot_idx].clear()

                        if self.last_selection[0] is not False:
                            # recolor preselected points
                            selected_obs = x[self.last_selection][0]
                            other_obs = np.delete(
                                x, self.last_selection)
                            print(selected_obs.shape, other_obs.shape)
                            fc_sel = plot_dicts[subplot_idx]["fc"]
                            fc_sel[-1] = 1
                            fc_not_sel = fc_sel.copy()
                            fc_not_sel[-1] = alpha_other

                            color_map = [fc_sel, fc_not_sel]
                            hist = axs[subplot_idx].hist(
                                [selected_obs, other_obs],
                                stacked=True,
                                color=color_map)
                        else:
                            hist = axs[subplot_idx].hist(x)
                            axs[subplot_idx].set_box_aspect(aspect=1)
                            vlines = False
                            fc_sel = list(hist[2][0].get_facecolor())

                        axs[subplot_idx].set_box_aspect(aspect=1)
                        hist_variable_name = plot_object["obj"]
                        axs[subplot_idx].set_xlabel(hist_variable_name)
                        axs[subplot_idx].set_title(
                            f"Histogram of variable {hist_variable_name}")

                        plot_dict = {"type": "hist",
                                     "subtype": "hist",
                                     "ax": axs[subplot_idx],
                                     "data": x,
                                     "vlines": vlines,
                                     "fc": fc_sel}
                        plot_dicts[subplot_idx] = plot_dict
                        # start area selector
                        selector = SpanSelect(
                            plot_dicts=plot_dicts,
                            subplot_idx=subplot_idx,
                            alpha_other=alpha_other,
                            last_selection=self.last_selection)
                        selectors.append(selector)
                    else:
                        print("Column not found")

            self.pause_var = tk.StringVar()
            fig.canvas.mpl_connect("key_press_event", accept)
            parent.wait_variable(self.pause_var)

        parent.destroy()


def interactive_tour(data, col_names, plot_objects, half_range=None, n_max_cols=None):
    """Launch InteractiveTourInterface object"""
    root = tk.Tk()
    InteractiveTourInterface(root,
                             data,
                             col_names,
                             plot_objects,
                             half_range,
                             n_max_cols)
    root.mainloop()

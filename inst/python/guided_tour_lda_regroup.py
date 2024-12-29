import numpy as np
from functools import partial
import tkinter as tk
import customtkinter as ctk
import matplotlib as plt


def open_regroup_interface(parent):
    parent.regroup_interface = tk.Toplevel(parent)
    parent.regroup_interface.title("Guided tour - LDA - regroup")

    # Create a container frame to hold the sub-frames
    container = tk.Frame(parent.regroup_interface)
    container.pack(fill=tk.BOTH, expand=True)

    # Create a dictionary to store the switch variables by rows
    switch_vars = {i: [] for i in range(parent.n_subsets)}

    # First loop to create all IntVar instances and store them in switch_vars
    for i in range(parent.n_subsets + 1):
        for j in range(parent.n_subsets):
            var = tk.IntVar(value=1 if i == j + 1 else 0)
            switch_vars[j].append(var)
    parent.switch_vars = switch_vars

    # Second loop to create frames and switches
    for i in range(parent.n_subsets + 1):
        frame = tk.Frame(container, borderwidth=2, relief="solid")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        if i == 0:
            label = tk.Label(frame, text="Ignore")
            label.pack(padx=10, pady=10)
        else:
            label = tk.Label(frame, text=f"New subgroup {i}")
            label.pack(padx=10, pady=10)

        for j in range(parent.n_subsets):
            switch = ctk.CTkSwitch(
                frame, text=parent.subset_names[j].get(),
                variable=switch_vars[j][i],
                text_color=plt.colors.to_hex(
                    parent.colors[j], keep_alpha=False),
                command=partial(switch_command,
                                parent=parent,
                                row_index=j,
                                column_index=i))
            switch.pack(padx=10, pady=10)

    run_tour_button = ctk.CTkButton(
        parent.regroup_interface, text="Run tour", command=lambda: get_regrouping(parent))
    run_tour_button.pack(pady=20)


def switch_command(parent, row_index, column_index):
    for i, var in enumerate(parent.switch_vars[row_index]):
        if column_index != i:
            var.set(0)


def get_regrouping(parent):
    subselections = [np.array(subselection, dtype=np.float64)
                     for subselection in parent.subselections]

    grid = np.zeros((parent.n_subsets, parent.n_subsets + 1), dtype=int)
    for i in range(parent.n_subsets):
        for j in range(parent.n_subsets + 1):
            grid[i, j] = parent.switch_vars[i][j].get()

    cluster_array = np.full(parent.data.shape[0], -1)

    for i, subselection in enumerate(subselections):
        cluster_array[subselection.astype(int)] = i

    new_grouping = np.full(parent.data.shape[0], -1)

    for i, row in enumerate(grid.T):
        new_group = np.where(row == 1)
        new_grouping[np.in1d(cluster_array, new_group)] = i

    parent.new_grouping = new_grouping
    parent.regrouping_keep_obs = np.where(new_grouping != 0)[0]
    parent.wait_var.set(1)
    parent.regroup_interface.destroy()

import numpy as np

from helpers import gram_schmidt
from interactive_selectors import DraggableAnnotation1d, DraggableAnnotation2d, LassoSelect


def feature_checkbox_event(self, feature_idx):
    feature_selection = [state.get()
                         for state in self.feature_selection_vars]
    if sum(feature_selection) <= 1:
        self.feature_selection_vars[feature_idx].set(1)
    self.feature_selection = [state.get()
                              for state in self.feature_selection_vars]
    self.feature_selection = np.bool_(self.feature_selection)

    self.plot_loop()


def subselection_checkbox_event(self, subselection_idx):
    for i, _ in enumerate(self.subselection_vars):
        if i == subselection_idx:
            self.subselection_vars[i].set(1)
        else:
            self.subselection_vars[i].set(0)

        for plot_idx, _ in enumerate(self.plot_dicts):
            self.plot_dicts[plot_idx]["update_plot"] = False

    self.plot_loop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Any, Sequence
import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from .base import ArtifactHandler

# Use non-GUI backend
matplotlib.use('Agg')


class ObscCompareHandler(ArtifactHandler):
    """ Handler class to plot predicted vs. actual obscuration """

    def __init__(self):
        super().__init__()

        self.y_pred_l = []
        self.y_true_l = []

    def start(self, model: Any, dataloader: DataLoader) -> None:
        pass

    def on_iter(self, dl_iter: Sequence, model_out: Sequence) -> None:
        _, features = dl_iter
        features = features.cpu().detach().numpy()

        # Get prediction from model
        _, pred = model_out
        pred = torch.sigmoid(pred[..., -1])
        pred = pred.cpu().detach().numpy()

        # Get predicted label as argmax
        self.y_pred_l.append(pred)
        self.y_true_l.append(features[..., -1])

    def save(self, output_dir) -> pathlib.Path:
        y_pred = np.concatenate(self.y_pred_l) * 100.
        y_true = np.concatenate(self.y_true_l) * 100.

        # Create the plot!
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        ax.scatter(y_pred, y_true, 9)
        ax.set_xlabel(r'Predicted Obscuration [%]')
        ax.set_ylabel(r'Est. Obscuration by Logit Regression [%]')
        ax.set_title('Model Obscuration Prediction Accuracy')
        ax.grid()
        ax.plot((0, 100), (0, 100), '--k', linewidth=2)
        ax.legend(['Model Data', 'y = x'])

        output_path = output_dir / 'obsc_compare_plot.png'
        fig.savefig(str(output_path))
        plt.close(fig)

        return output_path

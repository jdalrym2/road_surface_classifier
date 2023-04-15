#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Any, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics

import torch
from torch.utils.data import DataLoader

from .base import ArtifactHandler

# Use non-GUI backend
matplotlib.use('Agg')


class AUCHandler(ArtifactHandler):
    """ Handler class to plot accuracy vs obscuration """

    def __init__(self):
        super().__init__()

        self.y_true_l = []     # true labels
        self.y_pred_c = []     # confidences
        self.roc_auc = None

    def start(self, model: Any, dataloader: DataLoader) -> None:
        pass

    def on_iter(self, dl_iter: Sequence, model_out: Sequence) -> None:

        _, features = dl_iter
        features = features.cpu().detach().numpy()

        _, pred = model_out
        pred = torch.softmax(pred, dim=1)
        pred = pred.cpu().detach().numpy()

        # Get predicted label as argmax
        this_y_true = np.argmax(features[..., :-1], axis=1)
        self.y_true_l.append(this_y_true)
        self.y_pred_c.append(pred[..., :-1])

    def save(self, output_dir) -> pathlib.Path:

        y_true_l = np.concatenate(self.y_true_l)
        # probability of the class with the *greater* / *positive* label
        y_pred_c = np.concatenate(self.y_pred_c)[:, 1]

        fig, ax = plt.subplots()
        ax.grid()
        disp = sklearn.metrics.RocCurveDisplay.from_predictions(y_true_l,
                                                                y_pred_c,
                                                                ax=ax)
        self.roc_auc = disp.roc_auc  # type: ignore
        ax.set_title('ROC Curve')

        plt_path = output_dir / 'roc_curve.png'
        fig.savefig(str(plt_path))
        plt.close(fig)

        return plt_path

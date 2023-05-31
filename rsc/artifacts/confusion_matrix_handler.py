#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Any, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .base import ArtifactHandler

# Use non-GUI backend
matplotlib.use('Agg')


class ConfusionMatrixHandler(ArtifactHandler):
    """ Handler class to generate a confusion matrix """

    def __init__(self):
        super().__init__()

        self.labels: list[str] | None = None
        self.y_true_l: list[Any] = []
        self.y_pred_l: list[Any] = []

    def start(self, model: Any, dataloader: DataLoader) -> None:

        # Try to get labels from model
        self.labels = model.__dict__.get('labels')

    def on_iter(self, dl_iter: Sequence, model_out: Sequence) -> None:

        # Parse iterable
        _, features = dl_iter

        # We extract just the image + location mask
        y_true = features.numpy()[..., :-1]
        self.y_true_l.append(y_true)

        # Get prediction from model
        _, pred = model_out
        pred = pred.cpu().detach().numpy()

        # Lazy-init labels if we couldn't get them from the model
        if self.labels is None:
            self.labels = [f'Class {n+1:d}' for n in range(len(pred))]

        # Get predicted label as argmax
        y_pred = np.argmax(pred[..., :-1], axis=1)
        self.y_pred_l.append(y_pred)

    def save(self, output_dir) -> pathlib.Path:

        # Aggregate and organize
        y_true = np.concatenate(self.y_true_l)
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.concatenate(self.y_pred_l)

        # Generate and save the confusion matrix
        c = ConfusionMatrixDisplay(confusion_matrix(y_true,
                                                    y_pred,
                                                    normalize='true'),
                                   display_labels=self.labels)
        output_path = output_dir / 'confusion_matrix.png'
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(left=0.2, bottom=0.2)
        try:
            c.plot(ax=ax, cmap=cm.Blues, # type: ignore
                   xticks_rotation='vertical')     
        except ValueError:
            print('Detected issue with plot! This is likely due to a mismatch of class labels and true labels.')
            raise
        fig.savefig(str(output_path))
        plt.close(fig)

        return output_path
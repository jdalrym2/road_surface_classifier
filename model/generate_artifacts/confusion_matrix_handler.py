#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from . import device
from .metrics_handler import MetricsHandler

# Use non-GUI backend
matplotlib.use('Agg')


class ConfusionMatrixHandler(MetricsHandler):

    def generate_artifact(self) -> pathlib.Path:
        """
        Plot a confusion matrix for a model.

        Args:
            model (Unknown): Preloaded model
            val_dl (DataLoader): Path to validation DataLoader to use
            output_path (pathlib.Path): Plot output path
            labels (Optional[List[str]], optional): Class labels. If not provided,
                will be attempted to be extracted from the model. Otherwise, non-specific
                values will be used. Defaults to None.
        """

        # Try to get labels
        labels = self.model.__dict__.get('labels')

        # Get truth and predictions using the dataloader
        y_true_l, y_pred_l = [], []
        for x, features in tqdm(iter(self.dataloader)):
            # We extract just the image + location mask
            x = x[:, 0:5, :, :].to(device)
            y_true_l.append(features.numpy())

            # Get prediction from model
            _, pred = self.model(x)
            pred = pred.cpu().detach().numpy()

            # Lazy-init labels if we couldn't get them from the model
            if labels is None:
                labels = [f'Class {n+1:d}' for n in range(len(pred))]

            # Get predicted label as argmax
            y_pred = np.argmax(pred, axis=1)
            y_pred_l.append(y_pred)

        # Aggregate and organize
        y_true = np.concatenate(y_true_l)
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.concatenate(y_pred_l)

        # Generate and save the confusion matrix
        c = ConfusionMatrixDisplay(confusion_matrix(y_true,
                                                    y_pred,
                                                    normalize='true'),
                                   display_labels=labels)
        output_path = self.output_dir / 'confusion_matrix.png'
        fig, ax = plt.subplots()
        c.plot(ax=ax, cmap=plt.cm.Blues)     # type: ignore
        fig.savefig(str(output_path))
        plt.close(fig)

        return output_path
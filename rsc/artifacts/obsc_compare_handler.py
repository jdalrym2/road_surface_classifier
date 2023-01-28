#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import device
from .metrics_handler import MetricsHandler

# Use non-GUI backend
matplotlib.use('Agg')


class ObscCompareHandler(MetricsHandler):

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

        # Get truth and predictions using the dataloader
        y_pred_l, y_true_l = [], []
        for x, features in tqdm(iter(self.dataloader)):
            features = features.cpu().detach().numpy()

            # We extract just the image + location mask
            x = x[:, 0:5, :, :].to(device)

            # Get prediction from model
            _, pred = self.model(x)
            pred = torch.softmax(pred, dim=1)
            pred = pred.cpu().detach().numpy()

            # Get predicted label as argmax
            y_pred_l.append(pred[..., 2])
            y_true_l.append(features[..., 2])

        y_pred = np.concatenate(y_pred_l) * 100.
        y_true = np.concatenate(y_true_l) * 100.

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

        output_path = self.output_dir / 'obsc_compare_plot.png'
        fig.savefig(str(output_path))
        plt.close(fig)

        return output_path
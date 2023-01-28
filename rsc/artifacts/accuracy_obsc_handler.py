#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import device
from .metrics_handler import MetricsHandler

# Use non-GUI backend
matplotlib.use('Agg')


class AccuracyObscHandler(MetricsHandler):

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
        y_true_l, acc_l, y_true_obsc_l = [], [], []
        for x, features in tqdm(iter(self.dataloader)):
            features = features.cpu().detach().numpy()

            # We extract just the image + location mask
            x = x[:, 0:5, :, :].to(device)

            # Get prediction from model
            _, pred = self.model(x)
            pred = pred.cpu().detach().numpy()

            # Lazy-init labels if we couldn't get them from the model
            if labels is None:
                labels = [f'Class {n+1:d}' for n in range(len(pred))]

            # Get predicted label as argmax
            this_y_pred = np.argmax(pred[..., :2], axis=1)
            this_y_true = np.argmax(features[..., :2], axis=1)
            y_true_l.append(this_y_true)
            acc_l.append((this_y_pred == this_y_true).astype(int))
            y_true_obsc_l.append(features[..., 2])

        # Aggregate and organize
        y_true = np.concatenate(y_true_l)
        acc = np.concatenate(acc_l)
        y_true_obsc = np.concatenate(y_true_obsc_l)

        # Bins for obscuration: 0 -> 1
        bins = np.linspace(0, 1, 20 + 1)

        # Compute accuracy scores
        acc_plt, counts = [], []
        for obsc_max in bins[1:]:
            idx = np.where(y_true_obsc <= obsc_max)[0]
            this_acc = acc[idx]
            num_total = len(this_acc)
            num_correct = np.count_nonzero(this_acc)

            this_y_true = y_true[idx]
            idx_paved = np.where(this_y_true == 0)[0]
            idx_unpaved = np.where(this_y_true == 1)[0]

            acc_all = num_correct / num_total
            acc_paved = np.count_nonzero(this_acc[idx_paved]) / len(idx_paved)
            acc_unpaved = np.count_nonzero(
                this_acc[idx_unpaved]) / len(idx_unpaved)
            acc_plt.append(
                [e * 100 for e in (acc_all, acc_paved, acc_unpaved)])

            count_paved = len(idx_paved) / len(idx)
            counts.append((count_paved * 100., (1 - count_paved) * 100.))

        # Create the plot!
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
        ax[0].plot(bins[1:] * 100, [a for a, _, _ in acc_plt],
                   'k-*',
                   linewidth=2)
        ax[0].plot(bins[1:] * 100, [(b, c) for _, b, c in acc_plt], '-*')
        ax[0].set_title('Prediction Accuracy vs. Obscuration')
        ax[0].set_ylim(None, 100)     # type: ignore
        ax[0].set_ylabel(r'Accuracy [%]')
        ax[0].grid()
        ax[0].legend(('All', 'Paved', 'Unpaved'))

        ax[1].plot(bins[1:] * 100, counts, '-*')
        ax[1].set_title('Label Proportion vs. Obscuration')
        ax[1].set_ylim(0, 100)
        ax[1].grid()
        ax[1].legend(['Paved', 'Unpaved'])
        ax[1].set_ylabel(f'Proportion [%]')
        ax[1].set_xlabel(r'Obscuration [%]')

        output_path = self.output_dir / 'acc_obsc_plot.png'
        fig.savefig(str(output_path))
        plt.close(fig)

        return output_path
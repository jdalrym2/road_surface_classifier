#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Any, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from .base import ArtifactHandler

# Use non-GUI backend
matplotlib.use('Agg')


class AccuracyObscHandler(ArtifactHandler):
    """ Handler class to plot accuracy vs obscuration """

    def __init__(self):
        super().__init__()

        self.y_true_l = []
        self.acc_l = []
        self.y_true_obsc_l = []

    def start(self, model: Any, dataloader: DataLoader) -> None:
        pass

    def on_iter(self, dl_iter: Sequence, model_out: Sequence) -> None:

        _, features = dl_iter
        features = features.cpu().detach().numpy()

        _, pred = model_out
        pred = pred.cpu().detach().numpy()

        # Get predicted label as argmax
        this_y_pred = np.argmax(pred[..., :-1], axis=1)
        this_y_true = np.argmax(features[..., :-1], axis=1)
        self.y_true_l.append(this_y_true)
        self.acc_l.append((this_y_pred == this_y_true).astype(int))
        self.y_true_obsc_l.append(features[..., -1])

    @staticmethod
    def compute_scores(
            acc: np.ndarray,
            y_true: np.ndarray) -> tuple[float, float, float, float]:
        """
        Compute accuracy scores and relevant data for plotting.

        Args:
            acc (np.ndarray): Boolean array where True -> correct result
            y_true (np.ndarray): The true label array. Must be the same size as `acc`

        Returns:
            tuple[float, float, float, float]: Output accuracy scores and relevant data. All in range [0, 1].
              - `acc_all`: Accuracy over all data in `y_true`
              - `acc_paved`: Accuracy for paved roads in `y_true`
              - `acc_unpaved`: Accuracy for unpaved roads in `y_true`
              - `pc_paved`: Proportion of paved roads in `y_true`
              NOTE: `pc_unpaved == 1 - pc_paved`
        """
        assert len(acc) == len(y_true)

        # Count total + num correct
        num_total = len(acc)
        num_correct = np.count_nonzero(acc)

        # Get indices for paved / unpaved labels
        idx_paved = np.where(y_true == 0)[0]
        idx_unpaved = np.where(y_true == 1)[0]

        # Compute accuracy scores
        acc_all = num_correct / num_total
        acc_paved = np.count_nonzero(acc[idx_paved]) / len(idx_paved)
        acc_unpaved = np.count_nonzero(acc[idx_unpaved]) / len(idx_unpaved)

        # Compute paved proportion
        pc_paved = len(idx_paved) / len(acc)

        return acc_all, acc_paved, acc_unpaved, pc_paved

    def save(self, output_dir) -> tuple[pathlib.Path, pathlib.Path]:

        # Aggregate and organize
        y_true = np.concatenate(self.y_true_l)
        acc = np.concatenate(self.acc_l)
        y_true_obsc = np.concatenate(self.y_true_obsc_l)

        # Bins for obscuration: 0 -> 1
        bins = np.linspace(0, 1, 20 + 1)

        # Compute accuracy scores in bins
        acc_plt_b, counts_b = [], []     # binned
        acc_plt_c, counts_c = [], []     # cumulative
        for obsc_min, obsc_max in zip(bins[:-1], bins[1:]):
            # Get indices of interest (indices in bin + cumulative)
            idx_b = np.where((obsc_min < y_true_obsc)
                             & (y_true_obsc <= obsc_max))[0]
            idx_c = np.where(y_true_obsc <= obsc_max)[0]

            # Compute accuracy scores and add to plot data:
            # Binned
            acc_all, acc_paved, acc_unpaved, count_paved = self.compute_scores(
                acc[idx_b], y_true[idx_b])
            acc_plt_b.append(
                [e * 100 for e in (acc_all, acc_paved, acc_unpaved)])
            counts_b.append((count_paved * 100., (1 - count_paved) * 100.))
            # Cumulative
            acc_all, acc_paved, acc_unpaved, count_paved = self.compute_scores(
                acc[idx_c], y_true[idx_c])
            acc_plt_c.append(
                [e * 100 for e in (acc_all, acc_paved, acc_unpaved)])
            counts_c.append((count_paved * 100., (1 - count_paved) * 100.))

        # Create the plots!
        # Binned
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
        ax[0].plot(bins[1:] * 100, [a for a, _, _ in acc_plt_b],
                   'k-*',
                   linewidth=2)
        ax[0].plot(bins[1:] * 100, [(b, c) for _, b, c in acc_plt_b], '-*')
        ax[0].set_title('Prediction Accuracy vs. Obscuration (Binned)')
        ax[0].set_ylim(None, 100)     # type: ignore
        ax[0].set_ylabel(r'Accuracy [%]')
        ax[0].grid()
        ax[0].legend(('All', 'Paved', 'Unpaved'))
        ax[1].plot(bins[1:] * 100, counts_b, '-*')
        ax[1].set_title('Label Proportion vs. Obscuration (Binned)')
        ax[1].set_ylim(0, 100)
        ax[1].grid()
        ax[1].legend(['Paved', 'Unpaved'])
        ax[1].set_ylabel(f'Proportion [%]')
        ax[1].set_xlabel(r'Obscuration [%]')
        binned_path = output_dir / 'acc_obsc_plot_binned.png'
        fig.savefig(str(binned_path))
        plt.close(fig)

        # Cumulative
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
        ax[0].plot(bins[1:] * 100, [a for a, _, _ in acc_plt_c],
                   'k-*',
                   linewidth=2)
        ax[0].plot(bins[1:] * 100, [(b, c) for _, b, c in acc_plt_c], '-*')
        ax[0].set_title('Prediction Accuracy vs. Obscuration (Cumulative)')
        ax[0].set_ylim(None, 100)     # type: ignore
        ax[0].set_ylabel(r'Accuracy [%]')
        ax[0].grid()
        ax[0].legend(('All', 'Paved', 'Unpaved'))
        ax[1].plot(bins[1:] * 100, counts_c, '-*')
        ax[1].set_title('Label Proportion vs. Obscuration (Cumulative)')
        ax[1].set_ylim(0, 100)
        ax[1].grid()
        ax[1].legend(['Paved', 'Unpaved'])
        ax[1].set_ylabel(f'Proportion [%]')
        ax[1].set_xlabel(r'Obscuration [%]')
        cumul_path = output_dir / 'acc_obsc_plot_cumul.png'
        fig.savefig(str(cumul_path))
        plt.close(fig)

        return binned_path, cumul_path

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
from typing import Any, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .base import ArtifactHandler

# Use non-GUI backend
matplotlib.use('Agg')


class SamplesHandler(ArtifactHandler):

    def __init__(self, max_per_category: int = 50):
        super().__init__()
        self.max_per_category = max_per_category
        self.labels = []
        self.samples = {}
        self.maxed_out = {}

    def start(self, model: Any, dataloader: DataLoader) -> None:
        # Try to get labels from model
        self.labels = model.__dict__.get('labels')[:-1]     # trim obsc label
        assert self.labels is not None     # TODO: have to have number of labels here?

        # Setup samples dictionary
        for correct in (True, False):
            self.samples[correct] = {}
            self.maxed_out[correct] = {}
            for label in self.labels:
                self.samples[correct][label] = []
                self.maxed_out[correct][label] = False

    def on_iter(self, dl_iter: Sequence, model_out: Sequence) -> None:

        # Skip all this extra processing if we reached our maximum sample count anyway
        for label in self.labels:
            if any(not self.maxed_out[correct][label]
                   for correct in (True, False)):
                break
        else:
            return     # we did not break, so we must be maxed out

        x, features = dl_iter

        # Get mask and image
        xpm = x[:, 5:6, :, :]
        x = x[:, 0:5, :, :]

        # Get true label
        y_true = features.cpu().detach().numpy()

        # Transform image and mask
        x_p = (np.moveaxis(x[:, 0:4, ...].numpy(), 1, -1) * 255.).astype(
            np.uint8)
        xm_p = np.moveaxis(x[:, 4:5, ...].numpy(), 1, -1)
        xpm_p = np.moveaxis(xpm.numpy(), 1, -1)

        # Parse model preduction
        m, y_pred = model_out
        m_p = np.moveaxis(m.cpu().detach().numpy(), 1, -1)

        # True label (argmax)
        y_true_am: np.ndarray = np.argmax(y_true[:, 0:-2], 1)

        # Predicted label (argmax)
        y_pred_am = torch.argmax(y_pred[:, 0:-2], 1)
        y_pred_am = y_pred_am.cpu().detach().numpy()     # type: ignore

        # Predicted obscuration (softmax)
        y_pred_sm = torch.softmax(y_pred, 1)
        y_pred_sm = y_pred_sm.cpu().detach().numpy()     # type: ignore

        for i in range(x.shape[0]):

            # Are we correct?
            correct = y_true_am[i] == y_pred_am[i]

            # Get current true label
            this_label = self.labels[y_true_am[i]]

            # Skip if above limit for category
            if len(self.samples[correct][this_label]) > self.max_per_category:
                self.maxed_out[correct][this_label] = True
                continue

            # If we got here, then take a sample
            self.samples[correct][this_label].append((
                x_p[i, ...],
                xm_p[i, ...],
                xpm_p[i, ...],
                m_p[i, ...],
                y_true[i, ...],
                y_true_am[i, ...],
                y_pred_am[i, ...],
                y_pred_sm[i, ...],
            ))

    def save(self, output_dir: pathlib.Path) -> pathlib.Path:

        # Create directory for samples
        samples_dir = output_dir / 'samples'
        samples_dir.mkdir(parents=False, exist_ok=True)

        for correct in (True, False):
            # Create directory for "correctness"
            correct_dir = samples_dir / ('correct' if correct else 'incorrect')
            correct_dir.mkdir(parents=False, exist_ok=True)

            for label in self.labels:
                # Create directory for label
                label_dir = correct_dir / str(label)
                label_dir.mkdir(parents=False, exist_ok=True)

                # Loop through all the samples
                for idx, (x_p, xm_p, xpm_p, m_p, \
                    y_true, y_true_am, y_pred_am, \
                        y_pred_sm) in enumerate(self.samples[correct][label]):

                    # Create figure
                    fig, ax = plt.subplots(1,
                                           5,
                                           sharex=True,
                                           sharey=True,
                                           figsize=(15, 3.5))
                    for _ax in ax:
                        _ax.xaxis.set_visible(False)
                        _ax.yaxis.set_visible(False)

                    # Plot the result, highlighting the road with the mask
                    ax[0].set_title(
                        'RGB\nTrue: %s; Pred: %s' %
                        (self.labels[y_true_am], self.labels[y_pred_am]))
                    ax[0].imshow(np.uint8(x_p[..., (0, 1, 2)]))

                    ax[1].set_title('Color IR\nObsc: %.1f%%; Pred: %.1f%%' %
                                    (y_true[2] * 100, y_pred_sm[2] * 100))
                    ax[1].imshow(np.uint8(x_p[..., (3, 0, 1)]))

                    ax[2].set_title('Combined Image \n+ Mask')
                    ax[2].imshow(
                        np.uint8(x_p[..., (0, 1, 2)] * (0.33 + 0.67 * xm_p)))

                    ax[3].set_title('Combined Image \n+ True ProbMask')
                    ax[3].imshow(np.uint8(0.5 * x_p[..., (0, 1, 2)]))
                    ax[3].imshow(xpm_p,
                                 cmap='magma',
                                 vmin=0,
                                 vmax=1,
                                 alpha=0.33)

                    ax[4].set_title('Combined Image \n+ Pred ProbMask')
                    ax[4].imshow(np.uint8(0.5 * x_p[..., (0, 1, 2)]))
                    ax[4].imshow(m_p, cmap='magma', vmin=0, vmax=1, alpha=0.33)

                    # Save the figure
                    output_path = label_dir / f'sample_{idx:05d}.png'
                    fig.savefig(str(output_path))
                    plt.close(fig)

        return samples_dir
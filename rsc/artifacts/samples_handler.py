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
        self.labels = model.__dict__.get('labels')
        assert self.labels is not None

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
        xpm = x[:, (-1,), :, :]
        xm = x[:, (-2,), :, :]
        x = x[:, :-2, :, :]

        # Get true label
        y_true = features.cpu().detach().numpy()

        # Transform image and mask
        x_p = (np.moveaxis(x.numpy(), 1, -1) * 255.).astype(
            np.uint8)
        xm_p = np.moveaxis(xm.numpy(), 1, -1)
        xpm_p = np.moveaxis(xpm.numpy(), 1, -1)

        # Parse model prediction
        m, y_pred = model_out
        m_p = np.moveaxis(m.cpu().detach().numpy(), 1, -1)

        # True label (argmax)
        y_true_am: np.ndarray = np.argmax(y_true[:, 0:-1], 1)

        # Predicted label (argmax)
        y_pred_am = torch.argmax(y_pred[:, :-1], 1)
        y_pred_am = y_pred_am.cpu().detach().numpy()     # type: ignore

        # Predicted obscuration (sigmoid)
        y_pred_obsc = torch.sigmoid(y_pred[:, -1]).cpu().detach().numpy()  # type: ignore

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
                y_pred_obsc[i, ...],
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
                        y_pred_obsc) in enumerate(self.samples[correct][label]):
                    
                    # Get num-channels (3 - RGB, 4 - RGB + NIR)
                    _, _, nc = x_p.shape

                    # Create figure
                    fig, ax = plt.subplots(1,
                                           nc + 1,
                                           sharex=True,
                                           sharey=True,
                                           figsize=(15, 3.5))
                    for _ax in ax:
                        _ax.xaxis.set_visible(False)
                        _ax.yaxis.set_visible(False)

                    # Allows us to support different
                    # number of subplots in NIR case
                    ax_idx = 0

                    # Plot the result, highlighting the road with the mask
                    ax[ax_idx].set_title(
                        'RGB\nTrue: %s; Pred: %s' %
                        (self.labels[y_true_am], self.labels[y_pred_am]))
                    ax[ax_idx].imshow(np.uint8(x_p[..., (0, 1, 2)]))
                    ax_idx += 1

                    # Plot color IR if relevant
                    if nc > 3:
                        ax[ax_idx].set_title('Color IR')
                        ax[ax_idx].imshow(np.uint8(x_p[..., (3, 0, 1)]))
                        ax_idx += 1

                    ax[ax_idx].set_title('Combined Image + Mask\nObsc: %.1f%%; Pred: %.1f%%' % (y_true[-1] * 100, y_pred_obsc * 100))
                    ax[ax_idx].imshow(
                        np.uint8(x_p[..., (0, 1, 2)] * (0.33 + 0.67 * xm_p)))
                    ax_idx += 1

                    ax[ax_idx].set_title('Combined Image \n+ True ProbMask')
                    ax[ax_idx].imshow(np.uint8(0.5 * x_p[..., (0, 1, 2)]))
                    ax[ax_idx].imshow(xpm_p,
                                 cmap='magma',
                                 vmin=0,
                                 vmax=1,
                                 alpha=0.33)
                    ax_idx += 1

                    ax[ax_idx].set_title('Combined Image \n+ Pred ProbMask')
                    ax[ax_idx].imshow(np.uint8(0.5 * x_p[..., (0, 1, 2)]))
                    ax[ax_idx].imshow(m_p, cmap='magma', vmin=0, vmax=1, alpha=0.33)

                    # Save the figure
                    output_path = label_dir / f'sample_{idx:05d}.png'
                    fig.savefig(str(output_path))
                    plt.close(fig)

        return samples_dir
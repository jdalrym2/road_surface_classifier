#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from telnetlib import IP
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader


def sample_augmentations(ds, transform, grid=2):

    dl = DataLoader(ds, num_workers=1, batch_size=1, shuffle=True)
    for x, _ in dl:
        break
    x = x[0, ...]     # type: ignore
    x_np = x.numpy()
    im_np = np.moveaxis(x_np[:3, :, :], 0, -1).astype(np.uint8)
    mask_np = np.moveaxis(x_np[3:, :, :], 0, -1).astype(np.uint8)

    fig, ax = plt.subplots(grid * 2,
                           grid * 2,
                           sharex=True,
                           sharey=True,
                           figsize=(12, 12))
    ax = ax.flatten()     # type: ignore
    ax[0].imshow(im_np)
    ax[1].imshow(mask_np)

    # Plot images
    for idx in range(2, len(ax), 2):
        # Do an augmentation
        x_aug = transform(x)
        im_aug = x_aug[0, :3, ...]
        mask_aug = x_aug[0, 3:, ...]
        im_aug_np = (np.moveaxis(im_aug.numpy(), 0, -1) * 255.0).astype(
            np.uint8)
        mask_aug_np = np.moveaxis(mask_aug.numpy(), 0, -1).astype(np.uint8)

        # Plot it
        ax[idx].imshow(im_aug_np)
        ax[idx + 1].imshow(mask_aug_np)

    for _ax in ax:
        _ax.get_xaxis().set_visible(False)
        _ax.get_yaxis().set_visible(False)
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':

    from custom_image_dataset import RoadSurfaceDataset
    from data_augmentation import PreProcess, DataAugmentation

    train_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset_simple/dataset_train.csv',
        transform=PreProcess(),
        limit=50)

    sample_augmentations(train_ds, DataAugmentation())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


def sample_augmentations(ds, transform, grid=5):

    dl = DataLoader(ds, num_workers=1, batch_size=1, shuffle=True)
    for x, _ in dl:
        break
    x = x[0, ...]     # type: ignore
    x_np = x.numpy()
    im_np = np.moveaxis(x_np[0:4, :, :] * 255., 0, -1).astype(np.uint8)
    mask_np = np.moveaxis(x_np[4:5, :, :] * 255., 0, -1).astype(np.uint8)
    pmask_np = np.moveaxis(x_np[5:6, :, :] * 255., 0, -1).astype(np.uint8)

    fig, ax = plt.subplots(grid,
                           4,
                           sharex=True,
                           sharey=True,
                           figsize=(3 * 4, 3 * grid))
    ax = ax.flatten()     # type: ignore

    ax[0].imshow(im_np[..., (0, 1, 2)])     # type: ignore
    ax[1].imshow(im_np[..., (3, 0, 1)])     # type: ignore
    ax[2].imshow(mask_np)
    ax[3].imshow(pmask_np)

    # Plot images
    for idx in range(4, len(ax), 4):
        # Do an augmentation
        im_aug, pm_aug = transform(x)
        m_aug = im_aug[:, 4:5, ...]
        im_aug = im_aug[:, 0:4, ...]

        im_aug_np = (np.moveaxis(im_aug[0, ...].numpy(), 0, -1) * 255.).astype(
            np.uint8)
        m_aug_np = np.moveaxis(m_aug[0, ...].numpy() * 255., 0,
                               -1).astype(np.uint8)
        pm_aug_np = np.moveaxis(pm_aug[0, ...].numpy() * 255., 0,
                                -1).astype(np.uint8)

        # Plot it
        ax[idx].imshow(im_aug_np[..., (0, 1, 2)])     # type: ignore
        ax[idx + 1].imshow(im_aug_np[..., (3, 0, 1)])     # type: ignore
        ax[idx + 2].imshow(m_aug_np)
        ax[idx + 3].imshow(pm_aug_np)

    for _ax in ax:
        _ax.get_xaxis().set_visible(False)
        _ax.get_yaxis().set_visible(False)
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':

    from rsc.train.dataset import RoadSurfaceDataset
    from rsc.train.preprocess import PreProcess
    from rsc.train.data_augmentation import DataAugmentation

    train_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset/dataset_train.csv',
        transform=PreProcess(),
        limit=50)

    sample_augmentations(train_ds, DataAugmentation())
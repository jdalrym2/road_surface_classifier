#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any
import pandas as pd
import numpy as np
import PIL.Image

import torch
from torch.utils.data import Dataset


class RoadSurfaceDataset(Dataset):

    def __init__(self, df_path, transform, n_channels=4, limit=-1):

        # Load dataframe, interpret lenght
        self.df = pd.read_csv(df_path)
        self.n_idxs = len(self.df) if limit == -1 else min(limit, len(self.df))

        print(self.n_idxs)

        # Number of channels in the image
        self.n_channels = n_channels

        # Transformation object
        self.transform = transform

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):

        # Get row
        row: Any = self.df.iloc[idx]     # type: ignore

        # Image
        with PIL.Image.open(row.chip_path) as pim:
            im = np.array(pim)
        tn = im.shape[2] if im.ndim == 3 else 1
        if tn > self.n_channels:
            print(
                f'WARNING: Got {im.shape[2]} channel image but model only has {self.n_channels} dimensions!'
            )
            im = im[:, :, :self.n_channels]

        # Mask
        with PIL.Image.open(row.mask_path) as pmask:
            mask = np.array(pmask)
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        tn = im.shape[2]
        if tn > 1:
            mask = mask[:, :, 0][:, :, np.newaxis]

        # Prob mask
        with PIL.Image.open(row.probmask_path) as pmask:
            probmask = np.array(pmask)
        if probmask.ndim == 2:
            probmask = probmask[:, :, np.newaxis]
        tn = im.shape[2]
        if tn > 1:
            probmask = probmask[:, :, 0][:, :, np.newaxis]

        # Label
        lbl = [0] * 3     # FIXME: variable number of classes

        # Get class idx
        c = int(row.class_num)

        # Compute obscuration estimate
        obsc = 1 - (mask * (probmask > 127)).sum() / mask.sum()

        # Set fuzzy labels accordingly
        lbl[c] = 1 - obsc
        lbl[-1] = obsc

        # Concat image and masks for output
        x = self.transform(np.concatenate((im, mask, probmask), axis=2))

        return x, torch.Tensor(lbl)

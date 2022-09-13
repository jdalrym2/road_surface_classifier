#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import PIL.Image
from tqdm import tqdm

from torch.utils.data import Dataset


class RoadSurfaceDataset(Dataset):

    def __init__(self, df_path, transform, n_channels=3, limit=-1):
        df = pd.read_csv(df_path)
        self.transform = transform

        ims = []
        masks = []
        lbls = []

        print('Loading images from file...')
        n_idxs = len(df) if limit == -1 else min(limit, len(df))
        for idx in tqdm(range(n_idxs)):
            row = df.iloc[idx]
            with PIL.Image.open(row.chip_path) as pim:
                im = np.array(pim)
            tn = im.shape[2] if im.ndim == 3 else 1
            if tn > n_channels:
                im = im[:, :, :n_channels]

            with PIL.Image.open(row.mask_path) as pmask:
                mask = np.array(pmask)
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]
            tn = im.shape[2]
            if tn > 1:
                mask = mask[:, :, 0][:, :, np.newaxis]

            ims.append(im)
            masks.append(mask)
            lbls.append(int(row.class_num))

        combined = [
            np.concatenate((im, mask), axis=2) for im, mask in zip(ims, masks)
        ]

        self.stack = np.stack(combined, axis=0)
        self.lbls = lbls

    def __len__(self):
        return self.stack.shape[0]

    def __getitem__(self, idx):
        im, lbl = self.stack[idx, ...], self.lbls[idx]
        return self.transform(im), lbl

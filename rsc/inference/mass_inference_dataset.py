#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sqlite3
import pathlib
from typing import Any
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import torch.multiprocessing

from .fetch import fetch

torch.multiprocessing.set_sharing_strategy('file_system')

IMAGERY_PATH = pathlib.Path('/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019')
assert IMAGERY_PATH.is_dir()


class MassInferenceDataset(Dataset):
    def __init__(self,
                 sqlite_path: pathlib.Path,
                 transform,
                 n_channels=4,
                 limit=-1):

        # Connect to sqlite database and get all truthed roadways with no obscuration that are valid
        with sqlite3.connect('file:%s?mode=ro' % str(sqlite_path.resolve()),
                             uri=True) as con:
            self.df = pd.read_sql('SELECT * FROM features;',
                                  con).set_index('osm_id')

        # Load dataframe, interpret length
        self.n_idxs = len(self.df) if limit == -1 else min(limit, len(self.df))

        # Number of channels in the image
        self.n_channels = n_channels

        # Transformation object
        self.transform = transform

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):

        # Get row
        row: Any = self.df.iloc[idx]  # type: ignore

        x1, y1, x2, y2 = [row[e].item() for e in ('x1', 'y1', 'x2', 'y2')]

        # Mask
        im = fetch(IMAGERY_PATH / row['img'], x1, y1, x2, y2, row['wkt'])

        # Concat image and masks for output
        x = self.transform(im)

        return row.name, x

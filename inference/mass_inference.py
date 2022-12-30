#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore
""" Road Surface Classifier Inference Example
    This is a quick script to perform inference with our trained model, so we can see how well it works!
"""

# %%
# Hacky way to get PLMCNN in scope
import sys

sys.path.append('../model')

import pathlib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from mass_inference_dataset import MassInferenceDataset
from preprocess import PreProcess

# Input model and checkpoint paths (checkpoint contains the weights for inference)
# Since these files are so large, they are not in source control.
# Reach out to me if you'd like them.
model_path = pathlib.Path(
    '/data/road_surface_classifier/results/20221229_205449Z/model.pth')
assert model_path.exists()
ckpt_path = pathlib.Path(
    '/data/road_surface_classifier/results/20221229_205449Z/model-epoch=12-val_loss=0.41723.ckpt'
)
assert ckpt_path.exists()
ds_path = pathlib.Path('/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019.sqlite3')
assert ds_path.exists()

# %%
# Load model and checkpoint, set to eval (inference) mode
model = torch.load(model_path).load_from_checkpoint(ckpt_path)
model.eval()

# Get label array (it's built into model)
labels = model.__dict__.get('labels')

# Import dataset
preprocess = PreProcess()
val_ds = MassInferenceDataset(ds_path, transform=preprocess)
batch_size = 64
val_dl = DataLoader(val_ds,
                    num_workers=16,
                    batch_size=batch_size,
                    shuffle=True)

#%% Get truth and predictions using the dataloader

output = []

for i, (osm_id, x) in tqdm(enumerate(iter(val_dl)),
                           total=len(val_ds) // batch_size):

    sz = x.shape[0]

    # Get mask and image
    xm = x[:, 4:5, :, :]
    x = x[:, 0:4, :, :]

    # Predict with the model
    _, y_pred = model(x, xm)

    y_pred_am = torch.argmax(y_pred[:, 0:-1], dim=1)
    y_pred_am = y_pred_am.detach().numpy()     # type: ignore

    y_pred_sm = torch.softmax(y_pred, dim=1)
    y_pred_sm = y_pred_sm.detach().numpy()     # type: ignore

    for j in range(sz):
        output.append((int(osm_id[j]), labels[y_pred_am[j]], *y_pred_sm[j, :]))

#%%
import pandas as pd

columns = ['osm_id', 'pred_label', *['pred_%s' % label for label in labels]]

df = pd.DataFrame(output, columns=columns).set_index('osm_id')
df.to_csv(
    '/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019_results_20221229_205449Z.csv')

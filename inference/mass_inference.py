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
    '/data/road_surface_classifier/results/20230107_042006Z/model.pth')
assert model_path.exists()
ckpt_path = pathlib.Path(
    '/data/road_surface_classifier/results/20230107_042006Z/model-0-epoch=10-val_loss=0.39906.ckpt'
)
assert ckpt_path.exists()
ds_path = pathlib.Path('/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019.sqlite3')
assert ds_path.exists()

# %%
# Load model and checkpoint, set to eval (inference) mode
from plmcnn import PLMaskCNN

model = PLMaskCNN.load_from_checkpoint(ckpt_path)
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

    # Predict with the model
    _, y_pred = model(x)

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
    '/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019_results_20230107_042006Z.csv')

#%%
import sqlite3
import pandas as pd

df = pd.read_csv(
    '/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019_results_20230107_042006Z.csv'
).set_index('osm_id')

with sqlite3.connect(
        'file:/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019.sqlite3?mode=ro',
        uri=True) as con:
    df2 = pd.read_sql('SELECT * FROM features;', con).set_index('osm_id')

df = df.join(df2)
df

#%%
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

# Create SRS (EPSG:4326: WGS-84 decimal degrees)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

driver: ogr.Driver = ogr.GetDriverByName('GPKG')
ds: ogr.DataSource = driver.CreateDataSource(
    '/data/road_surface_classifier/BOULDER_COUNTY_NAIP_2019.gpkg')
layer: ogr.Layer = ds.CreateLayer('data', srs=srs, geom_type=ogr.wkbLineString)

osm_id_field = ogr.FieldDefn('osm_id', ogr.OFTInteger64)
highway_field = ogr.FieldDefn('highway', ogr.OFTString)
surface_true_field = ogr.FieldDefn('surface_t', ogr.OFTString)
surface_pred_field = ogr.FieldDefn('surface_p', ogr.OFTString)
paved_conf = ogr.FieldDefn('paved_c', ogr.OFSTFloat32)
unpaved_conf = ogr.FieldDefn('unpaved_c', ogr.OFSTFloat32)
obsc_conf = ogr.FieldDefn('obsc_c', ogr.OFSTFloat32)

layer.CreateField(osm_id_field)
layer.CreateField(highway_field)
layer.CreateField(surface_true_field)
layer.CreateField(surface_pred_field)
layer.CreateField(paved_conf)
layer.CreateField(unpaved_conf)
layer.CreateField(obsc_conf)

feature_defn = layer.GetLayerDefn()

for osm_id, row in df.iterrows():

    poly = ogr.CreateGeometryFromWkt(row['wkt'])

    feat = ogr.Feature(feature_defn)

    feat.SetGeometry(poly)
    feat.SetField('osm_id', row.name)
    feat.SetField('highway', row['highway_tag'])
    feat.SetField('surface_t', row['surface_tag'])
    feat.SetField('surface_p', row['pred_label'])
    feat.SetField('paved_c', row['pred_paved'])
    feat.SetField('unpaved_c', row['pred_unpaved'])
    feat.SetField('obsc_c', row['pred_Obscured'])

    layer.CreateFeature(feat)
    poly = None
    feat = None

layer = None     # type: ignore
ds = None     # type: ignore

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore
""" Road Surface Classifier Inference Example
    This is a quick script to perform inference with our trained model, so we can see how well it works!
"""

# %%

import pathlib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from rsc.inference.mass_inference_dataset import MassInferenceDataset
from rsc.model.preprocess import PreProcess

# Input model and checkpoint paths (checkpoint contains the weights for inference)
# Since these files are so large, they are not in source control.
# Reach out to me if you'd like them.
ckpt_path = pathlib.Path(
    '/data/road_surface_classifier/results/20230107_042006Z/model-0-epoch=10-val_loss=0.39906.ckpt'
)
assert ckpt_path.exists()
results_name = ckpt_path.parent.name
ds_path = pathlib.Path('/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019.sqlite3')
assert ds_path.exists()

# %% Setup processing chain for inference
from rsc.model.plmcnn import PLMaskCNN

# Load model and set to eval mode
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

#%% Iterate over the dataloader, and fetch predictions

# Output data array
output = []

for i, (osm_id, x) in tqdm(enumerate(iter(val_dl)),
                           total=len(val_ds) // batch_size):

    # Get size of this batch
    sz = x.shape[0]

    # Predict with the model
    _, y_pred = model(x)

    # Compute argmax
    y_pred_am = torch.argmax(y_pred[:, 0:-1], dim=1)
    y_pred_am = y_pred_am.detach().numpy()     # type: ignore

    # Compute softmax
    y_pred_sm = torch.softmax(y_pred, dim=1)
    y_pred_sm = y_pred_sm.detach().numpy()     # type: ignore

    for j in range(sz):
        output.append((int(osm_id[j]), labels[y_pred_am[j]], *y_pred_sm[j, :]))

#%% Save output as CSV
import pandas as pd

columns = ['osm_id', 'pred_label', *['pred_%s' % label for label in labels]]

df = pd.DataFrame(output, columns=columns).set_index('osm_id')
df.to_csv(
    f'/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019_results_{results_name}.csv')

#%% Load in CSV output, and merge with original dataset SQLite file
import sqlite3
import pandas as pd

df = pd.read_csv(
    f'/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019_results_{results_name}.csv'
).set_index('osm_id')

with sqlite3.connect(
        'file:/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019.sqlite3?mode=ro',
        uri=True) as con:
    df2 = pd.read_sql('SELECT * FROM features;', con).set_index('osm_id')

df = df.join(df2)
df

#%% Save results into GPKG file for import to QGIS
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

# Create SRS (EPSG:4326: WGS-84 decimal degrees)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

driver: ogr.Driver = ogr.GetDriverByName('GPKG')
ds: ogr.DataSource = driver.CreateDataSource(
    f'/data/road_surface_classifier/BOULDER_COUNTY_NAIP_2019_results_{results_name}.gpkg'
)
layer: ogr.Layer = ds.CreateLayer(
    'data', srs=srs, geom_type=ogr.wkbLineString)     # type: ignore

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

#%% Convert the CSV file into another QGIS GPKG, but pruned to ways with
#   known surface types, such that we can evaluate the model's accuracy easier
import sqlite3
import pandas as pd
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

# Read CSV, merge with dataset, extract the labels we want
df = pd.read_csv(
    f'/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019_results_{results_name}.csv'
).set_index('osm_id')
with sqlite3.connect(
        'file:/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019.sqlite3?mode=ro',
        uri=True) as con:
    df2 = pd.read_sql('SELECT * FROM features;', con).set_index('osm_id')
df = df.join(df2)
df = df[['wkt', 'pred_label', 'surface_tag', 'highway_tag', 'pred_Obscured']]

# Mapping of OSM surface types to simple paved / unpaved
class_map = {
    'asphalt': 'paved',
    'bricks': 'paved',
    'compacted': 'unpaved',
    'concrete': 'paved',
    'concrete:plates': 'paved',
    'dirt': 'unpaved',
    'gravel': 'unpaved',
    'ground': 'unpaved',
    'paved': 'paved',
    'paving_stones': 'paved',
    'unpaved': 'unpaved',
}

# Trim dataset + determine "correctness"
df = df[df['surface_tag'] != '']
df['surface_tag'] = df['surface_tag'].apply(class_map.get)
df['correct'] = df['surface_tag'] == df['pred_label']

# Create SRS (EPSG:4326: WGS-84 decimal degrees)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

# Put into GPKG format for QGIS
driver: ogr.Driver = ogr.GetDriverByName('GPKG')
ds: ogr.DataSource = driver.CreateDataSource(
    f'/data/road_surface_classifier/BOULDER_COUNTY_NAIP_2019_results_eval_{results_name}.gpkg'
)
layer: ogr.Layer = ds.CreateLayer('data', srs=srs, geom_type=ogr.wkbLineString)

osm_id_field = ogr.FieldDefn('osm_id', ogr.OFTInteger64)
highway_field = ogr.FieldDefn('highway', ogr.OFTString)
surface_true_field = ogr.FieldDefn('surface_t', ogr.OFTString)
surface_pred_field = ogr.FieldDefn('surface_p', ogr.OFTString)
correct_field = ogr.FieldDefn('correct', ogr.OFTString)
obsc_field = ogr.FieldDefn('obsc', ogr.OFTReal)

layer.CreateField(osm_id_field)
layer.CreateField(highway_field)
layer.CreateField(surface_true_field)
layer.CreateField(surface_pred_field)
layer.CreateField(correct_field)
layer.CreateField(obsc_field)

feature_defn = layer.GetLayerDefn()

for osm_id, row in df.iterrows():

    poly = ogr.CreateGeometryFromWkt(row['wkt'])

    feat = ogr.Feature(feature_defn)

    feat.SetGeometry(poly)
    feat.SetField('osm_id', row.name)
    feat.SetField('highway', row['highway_tag'])
    feat.SetField('surface_t', row['surface_tag'])
    feat.SetField('surface_p', row['pred_label'])
    feat.SetField('correct', str(row['correct']))
    feat.SetField('obsc', row['pred_Obscured'])

    layer.CreateFeature(feat)
    poly = None
    feat = None

layer = None     # type: ignore
ds = None     # type: ignore
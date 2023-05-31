#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Helper script used to inspect all available surface types within our dataset """
import pickle
import pathlib

from tqdm import tqdm
from osgeo import gdal

if __name__ == '__main__':

    feature_data = []
    feature_data_pickle_path = pathlib.Path(
        '/data/road_surface_classifier/feature_data.pkl')

    if not feature_data_pickle_path.exists():
        # Load road surface data from file
        ds = gdal.OpenEx(
            '/data/gis/us_road_surface/us_w_road_surface_filtered.gpkg')
        layer = ds.GetLayer()
        feature_count = layer.GetFeatureCount()

        # Extract features we care about
        print('Extracting features...')
        for idx in tqdm(range(feature_count)):
            feat = layer.GetNextFeature()
            wkt_str = feat.GetGeometryRef().ExportToWkt()
            osm_id = feat.GetField(0)
            highway = feat.GetField(1)
            surface = feat.GetField(2)
            feature_data.append((osm_id, wkt_str, highway, surface))
        layer = None
        ds = None

        # Pickle the data so we don't have to process again
        with open(feature_data_pickle_path, 'wb') as f:
            pickle.dump(feature_data, f)

    else:
        # Load the data from file
        with open(feature_data_pickle_path, 'rb') as f:
            feature_data = pickle.load(f)

    # Some data cleanup
    # Get all surface types with more than 1000 labels
    surface_types = list(set([e[3].lower() for e in feature_data]))
    count_dict = {k: 0 for k in surface_types}
    for _, _, _, surface in feature_data:
        count_dict[surface.lower()] += 1
    del_keys = [k for k, v in count_dict.items() if v < 1000]
    [count_dict.pop(k) for k in del_keys]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Pre-mass inference script. Loop through images and find chips of drivable ways. """
#%% Imports and inputs
import sqlite3

import pathlib
from osgeo import gdal, ogr, osr
import pandas as pd

from rsc.common.utils import imread_geometry, imread_dims, map_to_pix
from rsc.osm.overpass_api.road_network import OSMRoadNetworkOverpassQuery

gdal.UseExceptions()
ogr.UseExceptions()

OVERPASS_INTERPRETER_URL = "http://localhost:12345/api/interpreter"
TEST_CASE_PATH = pathlib.Path("/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019")
assert TEST_CASE_PATH.is_dir()
OUTPUT_PATH = pathlib.Path(
    '/nfs/taranis/naip/BOULDER_COUNTY_NAIP_2019.sqlite3')
assert not OUTPUT_PATH.exists()

# Setup custom query to local interpreter
q = OSMRoadNetworkOverpassQuery(format='xml', timeout=24 * 60 * 60)
q.set_endpoint('http://localhost:12345/api/interpreter')

# Get EPSG:4326 SRS
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

#%% Loop through imagery and find drivable ways

# Rows to convert to Pandas dataframe
df_rows = []

for img_path in TEST_CASE_PATH.glob('*.mrf'):
    print('Processing %s...' % img_path.name)

    # Get geometry for image
    img_geom = imread_geometry(img_path, wgs84=True)
    h, w = imread_dims(img_path)

    # Perform overpass query for geometry
    q.set_poly_from_wkt(img_geom)

    # Perform the query
    result = q.perform_query()

    # Convert to road network
    network = result.to_network()

    # If no roads show up, skip for now
    if not network.num_ways:
        continue

    # Compute image-specific coordinate transformation
    ds: gdal.Dataset = gdal.Open(str(img_path), gdal.GA_ReadOnly)
    im_h, im_w = ds.RasterYSize, ds.RasterXSize
    g_xform = ds.GetGeoTransform()
    srs_ds: osr.SpatialReference = ds.GetSpatialRef()
    srs_ds.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds = None     # type: ignore
    c_xform = osr.CoordinateTransformation(srs, srs_ds)

    # Loop through all available ways
    for way in network.get_ways():
        osm_id = way.id

        # Get nodes
        nodes = [network.get_node_by_id(id)
                 for id in way.nodes]     # type: ignore

        # Get linestring geometry
        linestr = ogr.Geometry(ogr.wkbLineString)
        [linestr.AddPoint_2D(node.lon, node.lat) for node in nodes]
        wkt = linestr.ExportToWkt()

        # Get midpoint lon, lat
        m_lon, m_lat = linestr.GetPoint_2D(linestr.GetPointCount() // 2)

        # Compute center point in image spatial coordinate system
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint_2D(m_lon, m_lat)
        pt.Transform(c_xform)

        # Compute upper-left corner for image chip
        x, y = [e[0].item() for e in map_to_pix(g_xform, pt.GetX(), pt.GetY())]
        if (x < 0 or x >= im_w) or (y < 0 or y >= im_h):
            # Center of way off of image, skipping.
            # NOTE: this OSM_ID may appear in another image
            continue
        x1, y1 = x - 128, y - 128
        x2, y2 = x1 + 256, y1 + 256

        # Handle boundaries
        if x1 < 0:
            x1, x2 = 0, 256
        if y1 < 0:
            y1, y2 = 0, 256
        if x2 >= w:
            x1, x2 = w - 256 - 1, w - 1
        if y2 >= h:
            y1, y2 = h - 256 - 1, h - 1

        # Fetch tags
        highway_tag = way.tags.get('highway', '')
        surface_tag = way.tags.get('surface', '')

        df_rows.append([
            osm_id, img_path.name, wkt, m_lon, m_lat, x1, y1, x2, y2,
            highway_tag, surface_tag
        ])

#%% Export data to SQLite3 database

df = pd.DataFrame(df_rows,
                  columns=[
                      'osm_id', 'img', 'wkt', 'm_lon', 'm_lat', 'x1', 'y1',
                      'x2', 'y2', 'highway_tag', 'surface_tag'
                  ]).set_index('osm_id')
with sqlite3.connect(OUTPUT_PATH) as con:
    df.to_sql('features', con)

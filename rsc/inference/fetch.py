#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from osgeo import ogr, osr
import PIL.Image
import PIL.ImageDraw

from rsc.common.utils import imread_geotransform, imread_srs, imread, map_to_pix

ogr.UseExceptions()

# Get ESPG:4326 reference SRS
srs_ref = osr.SpatialReference()
srs_ref.ImportFromEPSG(4326)
srs_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)


def fetch(img_path, x1, y1, x2, y2, wkt):

    # Fetch the tile we need
    h, w = y2 - y1, x2 - x1
    im = imread(str(img_path), x_off=x1, y_off=y1, w=w, h=h)
    xform = imread_geotransform(str(img_path), x_off=x1, y_off=y1)

    # Get the spatial reference (they change across the tiles)
    srs = osr.SpatialReference()
    srs.ImportFromProj4(imread_srs(str(img_path)))
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    im_trans = osr.CoordinateTransformation(srs_ref, srs)

    # Get list of points from WKT
    geom: ogr.Geometry = ogr.CreateGeometryFromWkt(wkt)
    geom.Transform(im_trans)
    pts = np.array(
        [geom.GetPoint_2D(idx) for idx in range(geom.GetPointCount())])

    # Convert to image-space x, y
    ix, iy = map_to_pix(list(xform), pts[:, 0], pts[:, 1])

    # Create a new image of the same shape, and draw a line
    # to create a mask
    mask_pil = PIL.Image.new('L', ((w, h)), color=0)
    d = PIL.ImageDraw.Draw(mask_pil)
    d.line(
        [(x, y) for x, y in zip(ix, iy)],     # type: ignore
        fill=255,
        width=2,
        joint="curve")
    mask = np.array(mask_pil)[:, :, np.newaxis]

    return np.concatenate((im, mask), axis=-1)
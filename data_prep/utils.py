#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Dataset Preparation Utilities """
from typing import Union, Tuple

import numpy as np
from osgeo import ogr, osr

ogr.UseExceptions()


def get_length_m(wkt_str: str) -> float:
    """
    Get length in meters of a WGS-84 linestring.

    NOTE:
    `srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)` is workaround
    for GDAL >= 3.0. See: https://github.com/OSGeo/gdal/issues/1546

    Args:
        wkt_str (str): WKT representation of linestring

    Returns:
        float: Length in meters
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    g = ogr.CreateGeometryFromWkt(wkt_str)
    srs2 = osr.SpatialReference()
    srs2.ImportFromEPSG(9834)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    trans = osr.CoordinateTransformation(srs, srs2)
    g.Transform(trans)

    return g.Length()


def wgs84_to_im(
        lon: Union[float, np.ndarray],
        lat: Union[float, np.ndarray],
        x: int,
        y: int,
        z: int,
        iw: int = 256,
        ih: int = 256
) -> Union[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    Convert WGS-84 coordinates (degrees) to image pixel values for a given
    tilemap tile and zoom (x, y, z)

    Args:
        lon (Union[float, np.ndarray]): Longitude or array of longitude values (degrees)
        lat (Union[float, np.ndarray]): Latitude or array of latitude values (degrees)
        x (int): Tilemap x-value
        y (int): Tilemap x-value
        z (int): Tilemap zoom level
        iw (int, optional): Tile width to compute output x-pixel. Defaults to 256.
        ih (int, optional): Tile height to compute output y-pixel. Defaults to 256.

    Returns:
        Union[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            Output tilemap values and pixel values: (tile x, tile y, pixel x, pixel y)
    """
    # Sanity check that inputs are the same type
    assert type(lon) == type(lat)

    # Convert everything to NumPy arrays to work with
    if not isinstance(lon, np.ndarray):
        lon_d, lat_d = np.array(lon), np.array(lat)
    else:
        lon_d, lat_d = lon, lat

    # Do the math! x, y -> "tile" x, y; ix, iy -> "image" xy
    n = np.power(2, z)
    ix_ar = (lon_d + 180.0) / 360.0 * n
    iy_ar = (1.0 - np.arcsinh(np.tan(np.deg2rad(lat_d))) / np.pi) / 2.0 * n
    ix_ar, iy_ar = np.floor(((ix_ar - x) * iw)).astype(int), np.floor(
        ((iy_ar - y) * ih)).astype(int)

    # Output array or integer values based on input datatype
    if not isinstance(lon, np.ndarray):
        ix: int = ix_ar.item()
        iy: int = iy_ar.item()
        return ix, iy
    else:
        return ix_ar, iy_ar
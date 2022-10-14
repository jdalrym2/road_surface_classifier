#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Dataset Preparation Utilities """
from typing import Union, Tuple, overload
import typing

import numpy as np
from osgeo import ogr, osr

ogr.UseExceptions()


@overload
def wgs84_to_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    z: int,
    iw: int = 256,
    ih: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...


@overload
def wgs84_to_xy(lon: float,
                lat: float,
                z: int,
                iw: int = 256,
                ih: int = 256) -> Tuple[int, int, int, int]:
    ...


def wgs84_to_xy(
    lon: Union[float, np.ndarray],
    lat: Union[float, np.ndarray],
    z: int,
    iw: int = 256,
    ih: int = 256
) -> Union[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray]]:
    """
    Convert WGS-84 coordinates (degrees) to tilemap x, y values for a
    given zoom level.

    Args:
        lon (Union[float, np.ndarray]): Longitude or array of longitude values (degrees)
        lat (Union[float, np.ndarray]): Latitude or array of latitude values (degrees)
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
    x_ar, y_ar = np.floor(ix_ar).astype(int), np.floor(iy_ar).astype(int)
    ix_ar, iy_ar = np.floor(((ix_ar - x_ar) * iw)).astype(int), np.floor(
        ((iy_ar - y_ar) * ih)).astype(int)

    # Output array or integer values based on input datatype
    if not isinstance(lon, np.ndarray):
        x: int = x_ar.item()
        y: int = y_ar.item()
        ix: int = ix_ar.item()
        iy: int = iy_ar.item()
        return x, y, ix, iy
    else:
        return x_ar, y_ar, ix_ar, iy_ar


@overload
def xy_to_wgs84(x: float, y: float, z: int) -> Tuple[float, float]:
    ...


@overload
def xy_to_wgs84(x: np.ndarray, y: np.ndarray,
                z: int) -> Tuple[np.ndarray, np.ndarray]:
    ...


def xy_to_wgs84(
        x: Union[float, np.ndarray], y: Union[float, np.ndarray],
        z: int) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Convert tile map x, y values to WGS-84 coordinates (degrees) for a given
    zoom level.

    Args:
        x (Union[float, np.ndarray]): Tilemap x-value or array of x-values
        y (Union[float, np.ndarray]): Tilemap y-value or array of y-values
        z (int): Tilemap zoom level

    Returns:
        Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]: Output
            WGS-84 coordinates: (lon, lat) (degrees)
    """
    # Sanity check that inputs are the same type
    assert type(x) == type(y)

    # Convert everything to NumPy arrays to work with
    if not isinstance(x, np.ndarray):
        x_ar, y_ar = np.array(x), np.array(y)
    else:
        x_ar, y_ar = x, y

    # Do the math!
    n = np.power(2, z)
    lon_d = x_ar / n * 360.0 - 180.0
    lat_d = np.rad2deg(np.arctan(np.sinh(np.pi * (1 - 2 * y_ar / n))))

    # Output array or integer values based on input datatype
    if not isinstance(x, np.ndarray):
        lon: float = lon_d.astype(float).item()
        lat: float = lat_d.astype(float).item()
        return lon, lat
    else:
        return lon_d, lat_d


@overload
def xy_to_bbox(x: float, y: float,
               z: int) -> Tuple[float, float, float, float]:
    ...


@overload
def xy_to_bbox(
        x: np.ndarray, y: np.ndarray,
        z: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...


def xy_to_bbox(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: int
) -> Union[Tuple[float, float, float, float], Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]]:
    """
    Convert tile map x, y values to WGS-84 coordinates (degrees) bounding box for a given
    zoom level.

    Args:
        x (Union[float, np.ndarray]): Tilemap x-value or array of x-values
        y (Union[float, np.ndarray]): Tilemap y-value or array of y-values
        z (int): Tilemap zoom level

    Returns:
        Union[Tuple[float, float, float, float], Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            Output WGS-84 coordinates: (lon1, lat1, lon2, lat2) (degrees)
    """
    # Sanity check that inputs are the same type
    assert type(x) == type(y)

    # Convert everything to NumPy arrays to work with
    if not isinstance(x, np.ndarray):
        x_ar, y_ar = np.array(x), np.array(y)
    else:
        x_ar, y_ar = x, y

    # Do the math; bbox is calculated by finding coordinates of x, y and x + 1, y + 1
    lon_1d, lat_1d = xy_to_wgs84(x_ar, y_ar, z)     # type: ignore
    lon_2d, lat_2d = xy_to_wgs84(x_ar + 1, y_ar + 1, z)     # type: ignore

    # Output array or integer values based on input datatype
    if not isinstance(x, np.ndarray):
        lon_1: float = lon_1d if isinstance(lon_1d, float) else lon_1d.item()
        lat_1: float = lat_1d if isinstance(lat_1d, float) else lat_1d.item()
        lon_2: float = lon_2d if isinstance(lon_2d, float) else lon_2d.item()
        lat_2: float = lat_2d if isinstance(lat_2d, float) else lat_2d.item()
        return lon_1, lat_1, lon_2, lat_2
    else:
        return lon_1d, lat_1d, lon_2d, lat_2d


@overload
def wgs84_to_im(lon: float,
                lat: float,
                x: int,
                y: int,
                z: int,
                iw: int = 256,
                ih: int = 256) -> Tuple[int, int]:
    ...


@overload
def wgs84_to_im(
    lon: np.ndarray,
    lat: np.ndarray,
    x: int,
    y: int,
    z: int,
    iw: int = 256,
    ih: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    ...


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
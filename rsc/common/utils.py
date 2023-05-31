#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Dataset Preparation Utilities """
import pathlib
import sqlite3
from typing import List, Optional, Union, Tuple, overload

import pandas as pd
import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
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


def map_to_pix(xform: list[float],
               x_m: Union[float, list[float], np.ndarray],
               y_m: Union[float, list[float], np.ndarray],
               round: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert x/y in map projection (WGS84: lon/lat) to pixel x, y coordinates

    Args:
        xform (list[float]): Geotransform to use.
        x_m (Union[float, list[float], np.ndarray]): Map x-coordinates in WGS84 (i.e. longitude)
        y_m (Union[float, list[float], np.ndarray]): Map y-coordinates in WGS84 (i.e. latitude)
        round (bool, optional): Whether or not to round the result and return integers.
            Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: Output x, y arrays in pixel coordinates.
    """

    # Input validation, either both float or both lists
    assert not (isinstance(x_m, float) ^ isinstance(y_m, float))
    if isinstance(x_m, float):
        x_m = [x_m]
    if isinstance(y_m, float):
        y_m = [y_m]
    x, y = np.array(x_m), np.array(y_m)
    assert x.ndim == y.ndim == 1
    assert len(x) == len(y)

    # Do the math!
    det = 1 / (xform[1] * xform[5] - xform[2] * xform[4])
    x_p = det * (xform[5] * (x - xform[0]) - xform[2] * (y - xform[3]))
    y_p = det * (xform[1] * (y - xform[3]) - xform[4] * (x - xform[0]))

    # Round pixel values, if desired
    if round:
        x_p, y_p = np.round(x_p).astype(int), np.round(y_p).astype(int)

    # Return
    return x_p, y_p


def pix_to_map(
    xform: list[float], x_p: Union[float, list[float], np.ndarray],
    y_p: Union[float, list[float],
               np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel x, y coordinates to x/y in map projection (WGS84: lon/lat)

    Args:
        xform (list[float]): Geotransform to use.
        x_p (Union[float, list[float], np.ndarray]): Pixel x-coordinates
        y_p (Union[float, list[float], np.ndarray]): Pixel y-coordinates

    Returns:
        tuple[np.ndarray, np.ndarray]: Output x, y arrays in map coordinates (WGS84: lon, lat).
    """
    # Input validation, either both float or both lists
    assert not (isinstance(x_p, float) ^ isinstance(y_p, float))
    if not isinstance(x_p, (list, tuple, np.ndarray)):
        x_p = [x_p]
    if not isinstance(y_p, (list, tuple, np.ndarray)):
        y_p = [y_p]
    x_p, y_p = np.array(x_p), np.array(y_p)
    assert x_p.ndim == y_p.ndim == 1
    assert len(x_p) == len(y_p)

    # Do the math!
    x_m = xform[0] + xform[1] * x_p + xform[2] * y_p
    y_m = xform[3] + xform[4] * x_p + xform[5] * y_p

    # Return
    return x_m, y_m


# Map of GDAL datatypes to numpy datatypes
GDAL_TO_NUMPY_MAP = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_UInt16: np.uint16,
    gdal.GDT_UInt32: np.uint32,
    gdal.GDT_Int16: np.int16,
    gdal.GDT_Int32: np.int32,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
    gdal.GDT_CInt16: np.complex128,
    gdal.GDT_CInt32: np.complex128,
    gdal.GDT_CFloat32: np.complex128,
    gdal.GDT_CFloat64: np.complex128,
    gdal.GDT_Unknown: np.float64,
}
NUMPY_TO_GDAL_MAP = dict([(v, k) for k, v in GDAL_TO_NUMPY_MAP.items()])

# Map of file extensions to GDAL formats
EXT_TO_FORMAT_MAP = {'.tif': 'GTiff', '.png': 'PNG'}


def imread(path: Union[str, pathlib.Path],
           x_off: int = 0,
           y_off: int = 0,
           w: Optional[int] = None,
           h: Optional[int] = None) -> np.ndarray:
    """
    Read in an image using GDAL

    Args:
        path (str): Path to image file.
        x_off (int, optional): X- offset to read image (top-left corner). Defaults to 0.
        y_off (int, optional): Y- offset to read image (top-left corner). Defaults to 0.
        w (Optional[int], optional): Width to read in. If None, defaults to the remainder of the image. Defaults to None.
        h (Optional[int], optional): Height to read in. If None, defaults to the remainder of the image. Defaults to None.

    Returns:
        np.ndarray: Loaded image, as a numpy array
    """
    # Open image as GDAL dataset, read only
    ds: gdal.Dataset = gdal.Open(str(path), gdal.GA_ReadOnly)

    # Ensure there are rasters to read
    assert ds.RasterCount > 0

    # Get the first band, figure out the datatype
    band: gdal.Band = ds.GetRasterBand(1)
    dt = GDAL_TO_NUMPY_MAP[band.DataType]

    # Determine image dimensions, and pre-allocate image
    im_w = w if w is not None else ds.RasterXSize - x_off
    im_h = h if h is not None else ds.RasterYSize - y_off
    im = np.empty((im_h, im_w, ds.RasterCount), dtype=dt)

    # Read the band
    im[:, :, 0] = band.ReadAsArray(x_off, y_off, w, h)
    band = None     # type: ignore

    # Read the remaining bands
    for ii in range(1, ds.RasterCount):
        band = ds.GetRasterBand(ii + 1)
        im[:, :, ii] = band.ReadAsArray(x_off, y_off, w, h)
        band = None     # type: ignore

    # Profit
    return im


def imread_geotransform(path: Union[str, pathlib.Path],
                        x_off: int = 0,
                        y_off: int = 0) -> Tuple:
    """
    Compute geotransform of a portion of an image

    Args:
        path (str): Path to image file.
        x_off (int, optional): X- offset in image (top-left corner). Defaults to 0.
        y_off (int, optional): Y- offset to read image (top-left corner). Defaults to 0.

    Returns:
        Tuple: Computed geotransform
    """
    # Open image as GDAL dataset, read only
    ds: gdal.Dataset = gdal.Open(str(path), gdal.GA_ReadOnly)

    # Get geotransform
    xform: Tuple = ds.GetGeoTransform()

    # Return for trival case
    if x_off == 0 and y_off == 0:
        return xform

    # Parse geotransform, we don't need the x- or y- offsets
    _, pw, rr, _, cr, ph = xform

    # Compute x / y offsets
    x_off, y_off = [e[0].item() for e in pix_to_map(list(xform), x_off, y_off)]

    # Return new geotransform
    return x_off, pw, rr, y_off, cr, ph


def imread_srs(path: Union[str, pathlib.Path]) -> str:
    """
    Get the spatial reference system for a raster given its path

    Args:
        path (Union[str, pathlib.Path]): Path to image

    Returns:
        srs: Spatial reference of the image as PROJ4 string
    """
    ds: gdal.Dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    srs: osr.SpatialReference = ds.GetSpatialRef()
    ds = None     # type: ignore
    return srs.ExportToProj4()


def imread_dims(path: Union[str, pathlib.Path]) -> Tuple[int, int]:
    """
    Get the dimensions a raster given its path

    Args:
        path (Union[str, pathlib.Path]): Path to image

    Returns:
        Tuple[int, int]: Image height, width
    """
    ds: gdal.Dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    h, w = ds.RasterYSize, ds.RasterXSize
    ds = None     # type: ignore
    return h, w


def imread_geometry(path: Union[str, pathlib.Path],
                    wgs84: bool = False) -> str:
    """
    Get the footprint a raster given its path.
    
    If wgs84 = False: The result is returned in the raster's native coordinate system.
    Else the result is return in ESPG:4326 (WGS-84) coordinates.

    Args:
        path (Union[str, pathlib.Path]): Path to image

    Returns:
        str: WKT string for the raster's geometry
    """
    # Get dimensions & geotransform
    h, w = imread_dims(path)
    xform = imread_geotransform(path)

    # Convert to map coordinates
    map_x, map_y = pix_to_map(list(xform), [0, w, w, 0, 0], [0, 0, h, h, 0])

    # Create geometry from coordinates
    poly = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    [ring.AddPoint_2D(x, y) for x, y in zip(map_x, map_y)]
    poly.AddGeometry(ring)

    # If desired, convert to WGS-84
    if wgs84:
        # Raster spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromProj4(imread_srs(path))
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # WGS-84 spatial reference
        srs_dst = osr.SpatialReference()
        srs_dst.ImportFromEPSG(4326)
        srs_dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # Transform!
        trans = osr.CoordinateTransformation(srs, srs_dst)
        poly.Transform(trans)

    # Return WKT-string for image footprint
    return poly.ExportToWkt()


def imwrite(im: np.ndarray,
            output_path: Union[str, pathlib.Path],
            xform: Optional[Tuple] = None,
            srs: Optional[str] = None) -> pathlib.Path:

    # Get output path
    output_path = pathlib.Path(output_path)
    if output_path.exists():
        raise FileExistsError('Path already exists! %s' %
                              str(output_path.resolve()))

    # Figure out dimensions to write
    assert im.ndim in (2, 3)
    im_h, im_w = im.shape[0], im.shape[1]
    im_c = 1 if im.ndim == 2 else im.shape[2]

    format = EXT_TO_FORMAT_MAP.get(output_path.suffix.lower())
    if format is None:
        raise ValueError('Unsupported format for extension: %s' %
                         output_path.suffix)
    dt = NUMPY_TO_GDAL_MAP.get(im.dtype.type)
    if dt is None:
        raise ValueError('Unsupported format for dtype: %s' % repr(im.dtype))
    options = ['COMPRESS=LZW'] if format == 'GTiff' else []

    try:
        mem_driver: gdal.Driver = gdal.GetDriverByName('MEM')
        mem_ds: gdal.Dataset = mem_driver.Create('',
                                                 im_w,
                                                 im_h,
                                                 im_c,
                                                 dt,
                                                 options=options)
        if xform is not None:
            mem_ds.SetGeoTransform(xform)
        if srs is not None:
            mem_ds.SetProjection(srs)

        for ii in range(im_c):
            band: gdal.Band = mem_ds.GetRasterBand(ii + 1)
            if im.ndim > 2:
                band.WriteArray(im[:, :, ii])
            else:
                band.WriteArray(im)
            band = None     # type: ignore

        out_driver: gdal.Driver = gdal.GetDriverByName(format)
        out_ds: gdal.Dataset = out_driver.CreateCopy(str(output_path), mem_ds)
        out_ds.FlushCache()
        out_ds = None     # type: ignore

    finally:
        mem_ds = None     # type: ignore

    # Return path we saved to
    return output_path


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


def pandas_load_sqlite(sqlite_path: Union[str, pathlib.Path],
                       table: str,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load a SQLite3 table into a Pandas dataframe.

    Args:
        sqlite_path (Union[str, pathlib.Path]): Path to sqlite3 file
        table (str): Table name to load into dataframe
        columns (Optional[List[str]], optional): Columns to load. Set to None to load all columns. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe representing sqlite3 table and appropriate columns
    """
    assert pathlib.Path(sqlite_path).exists()
    with sqlite3.connect(f'file:{str(sqlite_path)}?mode=ro', uri=True) as con:
        cursor = con.cursor()
        if columns is None:
            cursor.execute(f'PRAGMA table_info({table:s});')
            columns = [col for _, col, _, _, _, _ in cursor.fetchall()]
            cursor.execute(f"""
                SELECT * from {table:s};
            """)
        else:
            cursor.execute(f"""
                SELECT {', '.join(columns)} from {table:s};
            """)
    return pd.DataFrame(cursor.fetchall(), columns=columns)

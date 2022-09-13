#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Code to fetch USGS Imagery from nationalmap.gov """
import os
import sys
import pathlib
import urllib.parse
from typing import Tuple, Union

import requests
import numpy as np
from tqdm import tqdm


class NationalMapFetcher:
    """ Class to fetch USGS Imagery from nationalmap.gov """

    __slots__ = ['_db_loc']

    def __init__(self, db_loc: Union[str, pathlib.Path]):
        # Set filesystem database location
        self._db_loc = pathlib.Path(db_loc)
        if not self._db_loc.is_dir():
            raise ValueError('Database directory does not exist!')

    @staticmethod
    def wgs84_to_xy(
        lon: Union[float, np.ndarray],
        lat: Union[float, np.ndarray],
        z: int,
        iw: int = 256,
        ih: int = 256
    ) -> Union[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]]:
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

    @staticmethod
    def xy_to_wgs84(
            x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: int
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
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

    def fetch(self, z: int, x: int, y: int) -> pathlib.Path:
        """
        Fetch NationalMap imagery for a given set of
        tilemap coordinates.
        
        If the image is already in the filesystem database,
        this immediately returns the path to the image.
        Otherwise the image is fetched and its location in the
        database is returned.

        Args:
            z (int): Tilemap zoom level
            x (int): Tilemap x-value
            y (int): Tilemap y-value

        Raises:
            ValueError: If the URL fails to fetch (i.e. doesn't exist)

        Returns:
            pathlib.Path: Path to fetched image
        """
        # Determine output path
        output_path = self.get_save_path(z, x, y)
        if not output_path.exists():
            # Get fetch URL
            url = self.get_fetch_url(z, x, y)
            if not self.url_exists(url):
                raise ValueError('URL does not exist!')

            # Ensure parent directory exists by creating z and x
            # directories if they are not already there
            for p in (output_path.parents[1], output_path.parents[0]):
                p.mkdir(parents=False, exist_ok=True)

            # Do the thing!
            # TODO: make print configurable
            self.fetch_from_url(url, output_path, print=False)

        return output_path

    def get_save_path(self, z: int, x: int, y: int) -> pathlib.Path:
        """
        Get the desired filesystem location for a given image given
        its tilemap coordinates. This does not guarantee that the image
        exists: use `fetch` for that.

        Args:
            z (int): Tilemap zoom level
            x (int): Tilemap x-value
            y (int): Tilemap y-value

        Returns:
            pathlib.Path: Desired path to image
        """
        return self._db_loc / str(z) / str(y) / f'{x:d}.jpg'

    @staticmethod
    def get_fetch_url(z: int, x: int, y: int) -> str:
        """
        Get the nationalmap.gov fetch URL given a set of tilemap
        coordiantes.

        Args:
            z (int): Tilemap zoom level
            x (int): Tilemap x-value
            y (int): Tilemap y-value

        Returns:
            str: URL to fetch
        """
        return f'https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z:d}/{y:d}/{x:d}'

    @staticmethod
    def url_exists(url: str) -> bool:
        """
        Return if a URL exists and can be downloaded

        Args:
            url (str): URL to test

        Returns:
            bool: True if the URL exists, else False
        """
        response = requests.head(url)
        if response.status_code == 200:
            return 'content-length' in response.headers
        else:
            return False

    @staticmethod
    def fetch_from_url(url: str,
                       output_path: Union[str, pathlib.Path],
                       exist_ok: bool = False,
                       print: bool = True) -> pathlib.Path:
        """
        Fetch a file from a URL

        Args:
            url (str): URL to fetch
            output_path (Union[str, pathlib.Path]): Path to save fetched file. Can be
                a path to a directory or to a file.
            exist_ok (bool, optional): Whether or not it's okay that the file exists already.
                Defaults to False.
            print (bool, optional): Whether or not to print progress. Defaults to True.

        Raises:
            FileExistsError: If the file to save already exists.

        Returns:
            pathlib.Path: Path to saved file.
        """

        # Parse the incoming URL
        parse_result = urllib.parse.urlparse(url)

        # Resolve output path
        output_path = pathlib.Path(output_path).resolve()
        if not output_path.is_dir() and output_path.exists() and not exist_ok:
            raise FileExistsError('Output path already exists!')

        # If the output path was a directory, add a filename
        if output_path.is_dir():
            filename = pathlib.PosixPath(parse_result.path).name
            output_path = pathlib.Path(output_path, filename)

        # Fetch the file
        filesize = int(requests.head(url).headers["Content-Length"])
        with requests.get(url,
                          stream=True) as r, open(os.devnull, 'w') as dn, open(
                              output_path, 'wb') as f, tqdm(
                                  unit='B',
                                  unit_scale=True,
                                  unit_divisor=1024,
                                  total=filesize,
                                  file=sys.stdout if print else dn,
                                  desc=output_path.name) as progress:
            for chunk in r.iter_content(chunk_size=1024):
                size = f.write(chunk)
                progress.update(size)

        return output_path

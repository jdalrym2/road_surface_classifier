#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Classes to aid in OSM Overpass API Queries and Results """

from __future__ import annotations

import json
import pathlib
import textwrap
from abc import ABC, abstractmethod
from typing import Iterable, Union
import requests

from osgeo import ogr


class OSMOverpassQuery(ABC):
    """ Abstract class to perform an OSM Overpass API query """

    __slots__ = ['_endpoint', '_timeout', '_format', '_poly']

    def __init__(self, **kwargs):
        self._endpoint = kwargs.get(
            'endpoint', 'https://lz4.overpass-api.de/api/interpreter')
        self._timeout = kwargs.get('timeout', 180)
        self._format = kwargs.get('format', 'json')
        self._poly = None

    def set_poly_from_list(self, poly_list: Iterable[tuple[float,
                                                           float]]) -> None:
        """
        Set the polygon query boundary from a list of coordinates

        Args:
            poly_list (Iterable[tuple[float, float]]): List of floats in (lon, lat) format
        """
        self._poly = []
        for lon, lat in poly_list:
            self._poly.append((lon, lat))

    def set_poly_from_bbox(self, lat0: float, lon0: float, lat1: float,
                           lon1: float) -> None:
        """
        Set Overpass Query Filter polygon by bounding box

        Args:
            lat0 (float): First latitude (can be min or max)
            lon0 (float): First longitude (can be min or max)
            lat1 (float): Second latitude (can be min or max)
            lon1 (float): Second longitude (can be min or max)
        """
        # Find max, min latitude and longitude
        lat_max = max((lat0, lat1))
        lat_min = min((lat0, lat1))
        lon_max = max((lon0, lon1))
        lon_min = min((lon0, lon1))

        # Set polygon
        self._poly = [(lon_min, lat_max), (lon_max, lat_max),
                      (lon_max, lat_min), (lon_min, lat_min),
                      (lon_min, lat_max)]

    def set_poly_from_wkt(self, wkt_str: str) -> None:
        """
        Set Overpass Query Filter polygon by WKT string

        Args:
            wkt_str (str): Input polygon in WKT format

        Raises:
            ValueError: If the input WKT string is not of a polygon
        """
        # Load WKT string into JSON dict
        geojson_dict = json.loads(
            ogr.CreateGeometryFromWkt(wkt_str).ExportToJson())

        # Sanity check geometry type
        if geojson_dict['type'] != 'Polygon':
            raise ValueError('Input WKT string must be a polygon! Got: %s' %
                             geojson_dict['type'])

        # Set polygon (0 index is for linearRing inside polygon)
        self._poly = geojson_dict['coordinates'][0]

    @property
    def _poly_query_str(self) -> str:
        if self._poly is None:
            raise ValueError('Polygon not set!')
        return ' '.join([
            ' '.join((f'{lat:.6f}', f'{lon:.6f}')) for lon, lat in self._poly
        ])

    @property
    @abstractmethod
    def _query_str(self) -> str:
        return ''

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def set_endpoint(self, endpoint: str) -> None:
        """ Set the Overpass API Endpoint """
        self._endpoint = endpoint

    def _perform_query(self) -> requests.models.Response:
        """ Perform an OSM Overpass API Request! """
        query_str = textwrap.dedent(self._query_str).replace('\n', '')
        result = requests.get(self.endpoint, params={'data': query_str})
        result.raise_for_status()
        return result

    def perform_query(self) -> OSMOverpassResult:
        """ Perform an OSM Overpass API Request! """
        return OSMOverpassResult(self._perform_query())


class OSMOverpassResult:
    """ Container class for OSM Overpass result data. Will need to be subclassed to be useful. """
    __slots__ = ['_result']

    def __init__(self, result: requests.models.Response):
        self._result = result

    def to_file(self, output_path: Union[pathlib.Path, str]) -> None:
        """
        Have the query result to file. The file must be of the same format
        as the query result.

        Args:
            output_path (Union[pathlib.Path, str]): Output file path

        Raises:
            ValueError: if format is not recognized
        """
        # Get output path, sanity check dir
        output_path = pathlib.Path(output_path)
        assert output_path.parent.exists()

        # Assert output path suffix is valid
        this_format = self.format
        if 'json' in this_format:
            assert output_path.suffix.lower() == '.json'
        elif 'xml' in this_format:
            assert output_path.suffix.lower() in ('.xml', '.osm')
        else:
            raise ValueError('Unknown format: %s. Cannot export to file.' %
                             str(this_format))

        # Save to file!
        with open(output_path, 'wb') as f:
            f.write(self._result.content)

    @property
    def format(self) -> str:
        if 'Content-Type' in self._result.headers:
            return self._result.headers['Content-Type'].split('/')[-1]
        raise ValueError(
            'Unknown format! Content-Type not found in result header.')

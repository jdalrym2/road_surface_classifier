#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Code to fetch USGS Imagery from nationalmap.gov """

import urllib.parse

from .utils import xy_to_bbox
from .national_map_fetch import NationalMapFetcher


class NationalMapFetchNAIP(NationalMapFetcher):
    """ Class to fetch USGS Imagery from nationalmap.gov """

    @staticmethod
    def get_fetch_url(z: int, x: float, y: float) -> str:
        """
        Get the nationalmap.gov fetch URL given a set of tilemap
        coordiantes.

        Args:
            z (int): Tilemap zoom level
            x (float): Tilemap x-value (can be fractional)
            y (float): Tilemap y-value (can be fractional)

        Returns:
            str: URL to fetch
        """

        bbox = xy_to_bbox(x, y, z)

        query_args = {
            'f': 'image',
            'bbox': ','.join(['%.5f' % e for e in bbox]),
            'bboxSR': '4326',
            'size': '256,256',
            'imageSR': '3857',
            'format': 'jpgpng',
            'pixelType': 'U8',
            'interpolation': 'RSP_BilinearInterpolation',
            'adjustAspectRatio': 'true',
        }

        url = 'https://services.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage?'
        url += urllib.parse.urlencode(query_args)
        return url

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
            'f':
            'image',
            'bbox':
            ','.join(['%.5f' % e for e in bbox]),
            'bboxSR':
            '4326',
            'size':
            '256,256',
            'imageSR':
            '3857',
            'format':
            'jpgpng',
            'pixelType':
            'U8',
            'interpolation':
            'RSP_BilinearInterpolation',
            'adjustAspectRatio':
            'true',
            'renderingRule':
            r'{"rasterFunction":"FalseColorComposite"}',
            'token':
            r'13qgrEPT-KRNjF3sUOgHkhLrdNujmUXmS9OpOkNhdgkg7ZbZ_4ALw62UTw12C1AT8jrGYS2csEnXoV48fKa2fTJY7ymOuD1SUOPKjuoEAMvfzg4HLKaLm2ioJj7mENjdDNjZCYk3zA9CVRlkoHe0xf4hB6Ppi7df_0BTNEBpFNCy-c5GcF5AX84ZVdvAe5BShrYdsg9p7NKjDJVyXqCOwqGnD5PKENygOYbnO6fsp5ENIbuPRg1XsMx-9C6VGNu5tEpIrmdY45vVS8B7MWY3-RO1qRMjIHX-4EDj7a2N7lw.'
        }

        #url = 'https://services.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage?'
        url = 'https://naip.arcgis.com/arcgis/rest/services/NAIP/ImageServer/exportImage?'
        url += urllib.parse.urlencode(query_args)
        return url

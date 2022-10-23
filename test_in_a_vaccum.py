#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_prep.national_map_fetch_naip import NationalMapFetchNAIP
from data_prep.utils import wgs84_to_xy

if __name__ == '__main__':

    fetcher = NationalMapFetchNAIP(
        '/data/road_surface_classifier/national_map_ir')
    x, y, _, _ = wgs84_to_xy(-104.585, 39.055, 17)

    path = fetcher.fetch(17, x, y)
    print(path)

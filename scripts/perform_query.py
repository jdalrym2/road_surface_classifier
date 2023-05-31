#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" OSM Overpass query to find all drivable roads with a labeled 'surface' tag """
from __future__ import annotations
import pathlib

from rsc.osm.overpass_api import OSMOverpassQuery, OSMOverpassResult


class OSMCustomOverpassQuery(OSMOverpassQuery):
    """ Custom OSM Overpass API query for (hopefully) drivable road networks """

    __slots__ = ['_highway_tags']

    DEFAULT_HIGHWAY_TAGS = [
        'motorway', 'motorway_link', 'motorway_junction', 'trunk',
        'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link',
        'tertiary', 'tertiary_link', 'unclassified', 'residential'
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._highway_tags = kwargs.get('highway_tags',
                                        self.DEFAULT_HIGHWAY_TAGS)

    def perform_query(self) -> OSMCustomOverpassResult:
        """ Perform an OSM Overpass API Request! """
        return OSMCustomOverpassResult(self._perform_query())

    @property
    def _query_str(self) -> str:
        # NOTE: OSMNX uses instead
        # ["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|path|pedestrian|planned|platform|proposed|raceway|service|steps|track"]

        return f"""
            [out:{self._format}]
            [timeout:{self._timeout}]
            [maxsize:2147483648];
            (way["highway"]
            ["area"!~"yes"]
            ["access"!~"private"]
            ["highway"~"{'|'.join(self._highway_tags)}"]
            ["motor_vehicle"!~"no"]
            ["motorcar"!~"no"]
            ["surface"!~""]
            ["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]
            (poly:'{self._poly_query_str}');
            >;
            );
            out;
        """


class OSMCustomOverpassResult(OSMOverpassResult):
    """ Container class for OSM Overpass result data for (hopefully) drivable road networks """

    def to_file(self) -> None:
        """ Convert result data to an OSM network """
        tmp_file = pathlib.Path('/data/gis/result')
        this_format = self.format
        if 'json' in this_format:
            tmp_file = tmp_file.with_suffix('.json')
        elif 'xml' in this_format:
            tmp_file = tmp_file.with_suffix('.osm')
        else:
            raise ValueError(
                'Unknown format: \'%s\'! Cannot convert to OSM network.')

        # Save the response content to file
        tmp_file.write_bytes(self._result.content)


if __name__ == '__main__':

    # Setup custom query to local interpreter
    q = OSMCustomOverpassQuery(format='xml', timeout=24 * 60 * 60)
    q.set_endpoint('http://localhost:12345/api/interpreter')

    # Use rough USA bounds for query
    with open('gis/us_wkt.txt', 'r') as f:
        us_wkt = f.read()
    q.set_poly_from_wkt(us_wkt)

    # Perform query and save! This will take a long time.
    print('Performing query...')
    result = q.perform_query()
    print('Saving to file...')
    result.to_file()
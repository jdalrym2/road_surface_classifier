#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import tempfile
import pathlib

from ..osm_network import OSMNetwork
from .osm_overpass_api import OSMOverpassQuery, OSMOverpassResult


class OSMRoadNetworkOverpassQuery(OSMOverpassQuery):
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

    def perform_query(self) -> OSMRoadNetworkOverpassResult:
        """ Perform an OSM Overpass API Request! """
        return OSMRoadNetworkOverpassResult(self._perform_query())

    @property
    def _query_str(self) -> str:
        # NOTE: OSMNX uses instead
        # ["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|path|pedestrian|planned|platform|proposed|raceway|service|steps|track"]

        return f"""
            [out:{self._format}]
            [timeout:{self._timeout}];
            (way["highway"]
            ["area"!~"yes"]
            ["access"!~"private"]
            ["highway"~"{'|'.join(self._highway_tags)}"]
            ["motor_vehicle"!~"no"]
            ["motorcar"!~"no"]
            ["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]
            (poly:'{self._poly_query_str}');
            >;
            );
            out;
        """


class OSMRoadNetworkOverpassResult(OSMOverpassResult):
    """ Container class for OSM Overpass result data for (hopefully) drivable road networks """

    def to_network(self) -> OSMNetwork:
        """ Convert result data to an OSM network """
        with tempfile.TemporaryDirectory() as td:
            tmp_file = pathlib.Path(td, 'tmp')
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

            # Load the network!
            n = OSMNetwork.from_file(tmp_file)

        return n

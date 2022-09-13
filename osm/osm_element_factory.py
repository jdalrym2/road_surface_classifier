#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET

from .osm_element import OSMElement, OSMNode, OSMWay


class OSMElementFactory:
    """ Factory class for OSM elements """

    @staticmethod
    def from_json_dict(d: dict) -> OSMElement:
        # Read OSM 'type' key
        osm_type = d.get('type')

        # Create class
        if osm_type == 'node':
            return OSMNode.from_json_dict(d)
        elif osm_type == 'way':
            return OSMWay.from_json_dict(d)
        else:
            raise ValueError('Unknown OSM type: %s' % osm_type)

    @staticmethod
    def from_xml(el: ET.Element) -> OSMElement:
        # Read OSM 'type' key
        osm_type = el.tag

        # Create class
        if osm_type == 'node':
            return OSMNode.from_xml(el)
        elif osm_type == 'way':
            return OSMWay.from_xml(el)
        else:
            raise ValueError('Unknown XML tag: %s' % osm_type)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Overpass API Query Implementation """

from __future__ import annotations

import json
from typing import List, Union, TypeVar, Type
import warnings
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET

_gdal_available = False
try:
    from osgeo import ogr
    ogr.UseExceptions()
    _gdal_available = True
except ModuleNotFoundError:
    warnings.warn('Could not find GDAL. Some methods may not be available.', ImportWarning)

from . import gdal_required

# Generic type for abstract classmethods of OSMElement
T = TypeVar('T', bound='OSMElement')

class OSMElement(ABC):
    """ Abstract class to describe an OSM element (node, way, etc.) """
    __slots__ = ['id', 'tags']

    TYPE = ''     # holds OSM type string ('node', 'way', etc.)

    def __init__(self, **kwargs):
        """ Default constructor: sets attributes from kwargs """
        # Set id and tags (universal to every OSM element)
        # NOTE: it's okay if tags is empty, which is why it
        # doesn't get the fancy attribute wrapper
        self._set_attr_from_kwargs(kwargs, 'id', int, default=None)
        self.tags = kwargs.get('tags', {})

    def _set_attr_from_kwargs(self,
                              kwargs: dict,
                              attr_key: str,
                              dt: type,
                              default=None):
        """ Special way to set an attribute that throws an error if the default is used """
        try:
            self.__setattr__(attr_key, dt(kwargs[attr_key]))
        except (TypeError, ValueError):
            warnings.warn(
                f'Cannot cast input for \'{attr_key}\' to {dt}! Setting to {repr(default)}',
                RuntimeWarning)
            self.__setattr__(attr_key, default)
        except KeyError:
            warnings.warn(
                f'No {attr_key} specified in constructor! Setting to {repr(default)}',
                RuntimeWarning)
            self.__setattr__(attr_key, default)

    @staticmethod
    def _parse_xml_osm_tag(t: ET.Element) -> dict:
        """ Parse an XML OSM tag element into a dictionary """
        assert t.tag == 'tag'
        d = t.attrib
        if 'k' not in d or 'v' not in d:
            warnings.warn(f'Tag element empty! {d}', RuntimeWarning)
            return {}
        return {d['k']: d['v']}

    @staticmethod
    def _create_xml_osm_tag(t: dict) -> List[ET.Element]:
        """ Create a list of ETree Element objects from a tags dictionary """
        el_list = []     # type: List[ET.Element]
        for k, v in t.items():
            el = ET.Element('tag')
            el.attrib['k'] = str(k)
            el.attrib['v'] = str(v)
            el_list.append(el)
        return el_list

    def __str__(self):
        return f'<{self.__class__.__name__}; id: {self.id}; tags: {len(self.tags)}>'

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def to_json_dict(self) -> dict:
        """ Export this element to a JSON dict (Python dictionary) """
        return {'type': self.TYPE, 'id': self.id}

    @abstractmethod
    def to_xml(self) -> ET.Element:
        """ Export this element to an XML string """
        pass

    def to_xml_str(self) -> str:
        """ Export this element to an XML string """
        return ET.tostring(self.to_xml(), encoding='utf-8').decode('utf-8')

    @classmethod
    @abstractmethod
    def from_json_dict(cls: Type[T], json_dict: dict) -> T:
        """ Create this element from a JSON dict (Python dictionary) """
        pass

    def to_json(self) -> str:
        """ Export this element to an JSON string """
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """ Create this element from a JSON string """
        return cls.from_json_dict(json.loads(json_str))

    @classmethod
    @abstractmethod
    def from_xml(cls: Type[T], tree: ET.Element) -> T:
        """ Create this element from an XML Element """
        pass

    @classmethod
    def from_xml_str(cls: Type[T], xml_str: str) -> T:
        """ Create this element from an XML string """
        return cls.from_xml(ET.fromstring(xml_str))


class OSMNode(OSMElement):
    """ Class to describe an OSM node object """
    __slots__ = ['lat', 'lon']

    TYPE = 'node'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set latitude and longitude
        self._set_attr_from_kwargs(kwargs, 'lat', float, default=0.0)
        self._set_attr_from_kwargs(kwargs, 'lon', float, default=0.0)

    def __str__(self):
        s = super().__str__()[:-1]
        s += f'; lat: {self.lat:.3f}; lon: {self.lon:.3f}>'
        return s

    @gdal_required(_gdal_available)
    def to_ogr_geom(self) -> ogr.Geometry:
        """
        Convert this node to an OGR geometry object

        Note:
            Requires GDAL / OGR

        Returns:
            ogr.Geometry: OGR Geometry for node.
        """
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint_2D(self.lon, self.lat)
        return pt
    
    @gdal_required(_gdal_available)
    def to_wkt(self) -> str:
        """
        Get the WKT representation for this node.

        Note:
            Requires GDAL / OGR

        Returns:
            str: WKT representation for point
        """        
        return self.to_ogr_geom().ExportToWkt()

    def to_json_dict(self) -> dict:
        """ Export this element to a JSON dict (Python dictionary) """
        d = super().to_json_dict()
        d['lat'] = float(self.lat)
        d['lon'] = float(self.lon)
        if len(self.tags):
            d['tags'] = {}
            for k, v in self.tags.items():
                d['tags'][k] = str(v)
        return d

    def to_xml(self) -> ET.Element:
        """ Export this element to an XML string """
        # Create OSM XML element
        el = ET.Element('node')

        # Add id if it exists
        if self.id is not None:
            el.attrib['id'] = str(self.id)

        # Add any tags
        for tag_element in self._create_xml_osm_tag(self.tags):
            el.append(tag_element)

        # Add any remaining attributes
        el.attrib['lat'] = str(self.lat)
        el.attrib['lon'] = str(self.lon)

        # Return as XML Element
        return el

    @classmethod
    def from_json_dict(cls: Type[T], json_dict: dict) -> T:
        """ Create this element from a JSON dict (Python dictionary) """
        # Trick since the OSM keys match the attribute names exactly
        # i.e. 'id', 'lat', 'lon'
        # If this wasn't the case extra processing would be necessary
        return cls(**json_dict)

    @classmethod
    def from_xml(cls: Type[T], tree: ET.Element) -> T:
        """ Create this element from an XML string """
        # Assert OSM element type
        assert tree.tag == cls.TYPE

        # Trick since the OSM keys match the attribute names exactly
        # i.e. 'id', 'lat', 'lon'
        # If this wasn't the case extra processing would be necessary
        kwargs = tree.attrib.copy()

        # Parse any tags
        tags = {}
        for t in tree.findall('tag'):
            tags.update(cls._parse_xml_osm_tag(t))
        kwargs['tags'] = tags     # type: ignore

        return cls(**kwargs)


class OSMWay(OSMElement):
    """ Class to describe an OSM way object """
    __slots__ = ['nodes']

    TYPE = 'way'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set nodes
        self._set_attr_from_kwargs(kwargs, 'nodes', list, default=[])

    def __len__(self):
        return self.nodes.__len__()

    def __str__(self):
        s = super().__str__()[:-1]
        s += f'; nodes: {self.nodes.__len__()}>'
        return s

    @staticmethod
    def _parse_xml_osm_node(n: ET.Element) -> Union[int, None]:
        """ Parse an XML OSM tag element into a dictionary """
        assert n.tag == 'nd'
        d = n.attrib
        try:
            return int(d['ref'])
        except (TypeError, ValueError):
            warnings.warn(
                f'Node element id cannit be casted to int! Setting to None. {d["ref"]}',
                RuntimeWarning)
            return None
        except KeyError:
            warnings.warn(f'Node element empty! {d}', RuntimeWarning)
            return None

    @staticmethod
    def _create_xml_osm_node(n: int) -> ET.Element:
        """ Create a list of ETree Element objects from a node id """
        el = ET.Element('nd')
        el.attrib['ref'] = str(n)
        return el

    def to_json_dict(self) -> dict:
        """ Export this element to a JSON dict (Python dictionary) """
        d = super().to_json_dict()
        d['nodes'] = list(self.nodes)
        if len(self.tags):
            d['tags'] = {}
            for k, v in self.tags.items():
                d['tags'][k] = str(v)
        return d

    def to_xml(self) -> ET.Element:
        """ Export this element to an XML Element """
        # Create OSM XML element
        el = ET.Element('way')

        # Add id if it exists
        if self.id is not None:
            el.attrib['id'] = str(self.id)

        # Add any nodes
        for node in self.nodes:     # type: ignore
            el.append(self._create_xml_osm_node(node))

        # Add any tags
        for tag_element in self._create_xml_osm_tag(self.tags):
            el.append(tag_element)

        # Return as an element
        return el

    @classmethod
    def from_json_dict(cls: Type[T], json_dict: dict) -> T:
        """ Create this element from a JSON dict (Python dictionary) """
        # Trick since the OSM keys match the attribute names exactly
        # If this wasn't the case extra processing would be necessary
        return cls(**json_dict)

    @classmethod
    def from_xml(cls, tree: ET.Element) -> OSMWay:
        """ Create this element from an XML Element """
        # Assert OSM element type
        assert tree.tag == cls.TYPE

        # Kwargs to pass to class constructor
        kwargs = {}

        # Parse id
        if 'id' in tree.attrib:
            kwargs['id'] = tree.attrib['id']

        # Parse any tags
        nodes = []
        for nd in tree.findall('nd'):
            node_id = cls._parse_xml_osm_node(nd)
            if node_id is not None:
                nodes.append(node_id)
        kwargs['nodes'] = nodes     # type: ignore

        # Parse any tags
        tags = {}
        for t in tree.findall('tag'):
            tags.update(cls._parse_xml_osm_tag(t))
        kwargs['tags'] = tags     # type: ignore

        return cls(**kwargs)

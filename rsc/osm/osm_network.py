#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" OSMNetwork: Class to describe a network of OSM nodes and ways """

from __future__ import annotations

import json
import itertools
import pathlib
import tempfile
import warnings
from datetime import datetime
import xml.etree.ElementTree as ET
from typing import Iterator, List, Dict, Union, Optional

_gdal_available = False
try:
    from osgeo import gdal, ogr, osr
    _gdal_available = True
    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
except ModuleNotFoundError:
    warnings.warn('Could not find GDAL. Some methods may not be available.', ImportWarning)

from . import gdal_required

from . import get_logger
from .osm_element import OSMNode, OSMWay
from .osm_element_factory import OSMElementFactory

# OSM copyright and license
COPYRIGHT_STR = "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL."


class OSMNetwork:
    """ Class to describe a network of OSM nodes and ways """

    __slots__ = ['_nodes', '_ways', '_timestamp', '_generator']

    def __init__(self,
                 nodes_dict: Dict[int, OSMNode],
                 ways_dict: Dict[int, OSMWay],
                 generator: str = 'Unknown',
                 timestamp: Optional[str] = None,
                 copy: bool = True):
        """ Instantiate this class. This does not validate inputs!
            For input validation: use `from_dict` """
        # Input validation is performed if the user requests
        # Persist data, copy if needed
        self._nodes = nodes_dict
        self._ways = ways_dict
        self._generator = generator
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self._timestamp = timestamp     # type: str
        if copy:
            self._nodes = self._nodes.copy()
            self._ways = self._ways.copy()

    def __str__(self):
        return f'<{self.__class__.__name__}; nodes: {self.num_nodes}; ways: {self.num_ways}>'

    def __repr__(self):
        return self.__str__()
    
    def __iadd__(self, other: OSMNetwork):
        self._nodes.update(other._nodes)
        self._ways.update(other._ways)

    def get_node_by_id(self, node_id: int) -> OSMNode:
        """ Return an OSM node from its integer ID """
        try:
            return self._nodes[node_id]
        except KeyError:
            raise ValueError('Node ID was not found: %s' % repr(node_id))

    def get_way_by_id(self, way_id: int) -> OSMWay:
        """ Return an OSM way from its integer ID """
        try:
            return self._ways[way_id]
        except KeyError:
            raise ValueError('Way ID was not found: %s' % repr(way_id))
        
    @gdal_required(_gdal_available)
    def get_way_ogr_geometry(self, way_id: int) -> ogr.Geometry:      
        """
        Convert a way to an OGR geometry object

        Note:
            Requires GDAL / OGR

        Args:
            way_id (int): OSM Way ID

        Returns:
            ogr.Geometry: OGR Geometry for node.
        """
        # Look up node objects for way
        nodes = [self._nodes[n] for n in self._ways[way_id].nodes]

        # Construct OGR geometry
        geom = ogr.Geometry(ogr.wkbLineString)
        [geom.AddPoint_2D(n.lon, n.lat) for n in nodes]

        return geom
    
    @gdal_required(_gdal_available)
    def get_way_as_wkt(self, way_id: int) -> str:
        """
        Get the WKT representation for this node.

        Note:
            Requires GDAL / OGR

        Args:
            way_id (int): OSM Way ID

        Returns:
            str: WKT representation for point
        """ 
        return self.get_way_ogr_geometry(way_id).ExportToWkt()

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_ways(self) -> int:
        return len(self._ways)

    def get_nodes(self) -> Iterator[OSMNode]:
        """ Returns an iterator over the network's nodes """
        return iter(self._nodes.values())

    def get_ways(self) -> Iterator[OSMWay]:
        """ Returns an iterator over the network's ways """
        return iter(self._ways.values())

    def sort(self) -> None:
        """ Sort the nodes and ways in this network by ID """
        self._nodes = {k: self._nodes[k] for k in sorted(self._nodes.keys())}
        self._ways = {k: self._ways[k] for k in sorted(self._ways.keys())}

    def prune_nodes(self, sort=True) -> None:
        """
        Prune this network's nodes by removing any nodes
        that do not correlate to any existing ways.
        
        Note that it is not guaranteed the new nodes will be sorted
        unless sort = True.

        Args:
            sort (bool, optional): Whether or not to sort the list of nodes. Defaults to True.
        """
        # Get all of the nodes corresponding to all ways
        # in the network
        all_nodes = [way.nodes for way in self.get_ways()]

        # Flatten the list, and remove duplicates
        all_nodes = set(itertools.chain.from_iterable(all_nodes))

        # Sort, if we wish
        if sort:
            all_nodes = sorted(all_nodes)

        # Create a new node dictionary by taking only the nodes
        # we want
        nodes = {k: self.get_node_by_id(k) for k in all_nodes}

        # Replace this network's nodes dict
        self._nodes = nodes

    def clean_disjoint_regions(self, sort=True):
        """
        Clean up the network's disjoint regions by removing nodes
        and ways that are not connected to the network's largest
        contigious structure. May be used to sanitize the network
        after processing was done that may leave these disjoint
        regions.

        Note that it is not guaranteed the new nodes will be sorted
        unless sort = True.

        Args:
            sort (bool, optional): Whether or not to sort the list of nodes. Defaults to True.
        """
        # Find all nodes associated with more than one way
        # These dictate connections between ways and/or intersections
        node_dict = {node.id: [] for node in self.get_nodes()}
        for way in self.get_ways():
            for node in way.nodes:     # type: ignore
                node_dict[node].append(way.id)
        node_dict = {k: v for k, v in node_dict.items() if len(v) > 1}

        # Construct connectivity graph of ways based on the node_dict above
        way_dict = {}
        for node, way_list in node_dict.items():
            for way in way_list:
                if way not in way_dict:
                    way_dict[way] = []
                not_this = [k for k in way_list if k != way]
                way_dict[way].extend(not_this)
                way_dict.update({k: [] for k in not_this if k not in way_dict})

        # Apply Depth-First Search (DFS) algo to put ways into bins based
        # on their connectivity
        bins = {k: -1 for k in way_dict}
        bin_idx = 0
        for way in way_dict:
            if bins[way] == -1:
                stack = [way]
                while stack:
                    way = stack.pop()
                    if bins[way] >= 0:
                        continue
                    bins[way] = bin_idx
                    for neighbor in way_dict[way]:
                        stack.append(neighbor)
                bin_idx += 1

        # Count up the bins and determine the max
        count = {}
        for bin in bins.values():
            if bin not in count:
                count[bin] = 0
            count[bin] += 1
        max_bin = max(count, key=count.get)     # type: ignore

        # Reset ways based on which ones are in the max bin
        new_ways = {
            way_id: self.get_way_by_id(way_id)
            for way_id, v in bins.items() if v == max_bin
        }
        self._ways = new_ways

        # Prune orphaned nodes by this operation
        self.prune_nodes(sort=sort)

    def to_file(self,
                save_path: Union[str, pathlib.Path],
                format: str = 'json') -> None:
        """ Export the network to a file """
        # Cast save path to pathlib
        # and verify containing folder exists
        save_path = pathlib.Path(save_path)
        if not save_path.parent.exists():
            raise ValueError(
                f'Save directory does not exist: {str(save_path.parent)}')

        # Save in a variety of formats
        if format == 'json':
            return self._to_json(save_path)
        elif format == 'xml':
            return self._to_xml(save_path)
        elif format == 'gpkg':
            return self._to_gpkg(save_path)
        else:
            raise ValueError(f'Unknown format: {format}')

    @classmethod
    def from_file(cls, load_path: Union[str, pathlib.Path]):
        """ Import the network from a file """
        # Input validation
        load_path = pathlib.Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError('File path does not exist! %s' %
                                    str(load_path))
        ext = load_path.suffix.lower()
        if ext == '.json':
            return cls._from_json(load_path)
        elif ext in ('.xml', '.osm'):
            return cls._from_xml(load_path)
        elif ext == '.gpkg':
            return cls._from_gpkg(load_path)
        else:
            raise ValueError('File extension not supported: %s' % ext)

    @classmethod
    def from_list(cls,
                  nodes: List[OSMNode] = [],
                  ways: List[OSMWay] = [],
                  **kwargs) -> OSMNetwork:
        """ Instantiate the class from lists of OSM nodes and ways """
        # Dicts for filtered nodes and ways
        nodes_f = {}
        ways_f = {}

        # Input validation
        for node in nodes:
            if node.id is None:
                warnings.warn('Found node with ID of None. Skipping.')
                continue
            if not isinstance(node.id, int):
                raise ValueError('Node IDs must be integers! Found %s' %
                                 repr(node.id))
            if not issubclass(node.__class__, OSMNode):
                raise ValueError(
                    'Nodes must be inherited from OSMNode! Found %s' %
                    repr(node))
            nodes_f[node.id] = node
        for way in ways:
            if way.id is None:
                warnings.warn('Found way with ID of None. Skipping.')
                continue
            if not isinstance(way.id, int):
                raise ValueError('Way IDs must be integers! Found %s' %
                                 repr(way.id))
            if not issubclass(way.__class__, OSMWay):
                raise ValueError(
                    'Ways must be inherited from OSMWay! Found %s' % repr(way))
            ways_f[way.id] = way

        # Create the object!
        return cls(nodes_dict=nodes_f, ways_dict=ways_f, copy=False, **kwargs)

    @classmethod
    def from_dict(cls,
                  nodes: Dict[int, OSMNode] = {},
                  ways: Dict[int, OSMWay] = {},
                  **kwargs) -> OSMNetwork:
        """ Instantiate the class from a set of dicts, performing additional input validation """
        # Dicts for filtered nodes and ways
        nodes_f = {}
        ways_f = {}

        # Input validation
        for node_id, node in nodes.items():
            if node_id is None:
                warnings.warn('Found node with ID of None. Skipping.')
                continue
            if not isinstance(node_id, int):
                raise ValueError('Node IDs must be integers! Found %s' %
                                 repr(node_id))
            if not issubclass(node.__class__, OSMNode):
                raise ValueError(
                    'Nodes must be inherited from OSMNode! Found %s' %
                    repr(node))
            nodes_f[node_id] = node
        for way_id, way in ways.items():
            if way_id is None:
                warnings.warn('Found way with ID of None. Skipping.')
                continue
            if not isinstance(way_id, int):
                raise ValueError('Way IDs must be integers! Found %s' %
                                 repr(way_id))
            if not issubclass(way.__class__, OSMWay):
                raise ValueError(
                    'Ways must be inherited from OSMWay! Found %s' % repr(way))
            ways_f[way_id] = way

        # Create the object!
        return cls(nodes_dict=nodes_f, ways_dict=ways_f, copy=False, **kwargs)

    def _to_json(self, save_path: Union[str, pathlib.Path]) -> None:
        """ Save this network as an JSON file """
        # Create JSON dictionary
        # Contains generator, timestamp, and copyright info
        d = {
            'version': 0.6,
            'generator': self._generator,
            'osm3s': {
                'timestamp_osm_base': self._timestamp,
                'copyright': COPYRIGHT_STR
            },
            'elements': []
        }
        el = d['elements']     # hold pointer for speed

        # Append nodes
        for node in self._nodes.values():
            el.append(node.to_json_dict())

        # Append ways
        for way in self._ways.values():
            el.append(way.to_json_dict())

        # Save as file
        with open(save_path, 'w') as f:
            json.dump(d, f, indent=2)

    def _to_xml(self, save_path: Union[str, pathlib.Path]) -> None:
        """ Save this network as an XML file """
        # Create root OSM tag
        root = ET.Element('osm')
        root.attrib = dict(version='0.6', generator=self._generator)

        # Add OSM disclosure and license
        ET.SubElement(root, 'note').text = COPYRIGHT_STR     # type: ignore

        # Add timestamp metadata
        ET.SubElement(root, 'meta', dict(osm_base=self._timestamp))

        # Append nodes
        for node in self._nodes.values():
            root.append(node.to_xml())

        # Append ways
        for way in self._ways.values():
            root.append(way.to_xml())

        # Save as file
        tree = ET.ElementTree(root)
        tree.write(save_path, encoding='utf-8', xml_declaration=True)

    @gdal_required(_gdal_available)
    def _to_gpkg(self, save_path: Union[str, pathlib.Path]) -> None:
        """ Save this network as an GPKG file

        NOTE: in most use cases, the nodes and ways will be sorted by ID.
        This is assumed here. If it isn't the case, OGR may throw an error:

        ERROR 1: Non increasing node id. Use OSM_USE_CUSTOM_INDEXING=NO

        If this happens, call .sort() and try again or set OSM_USE_CUSTOM_INDEXING=NO.
        Both have various performance costs.

        See: https://gdal.org/drivers/vector/osm.html#internal-working-and-performance-tweaking

        """
        # Convert save_path to pathlib object
        save_path = pathlib.Path(save_path)

        # Add nodes list to way tags so we don't lose the node IDs upon import into OGR
        for way in self._ways.values():
            way.tags['osm_nodes'] = json.dumps(way.nodes)

        # Create SRS (EPSG:4326: WGS-84 decimal degrees)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # Save first as OSM and let OGR do the conversion to GPKG
        # Then, open the GPKG and add additional metadata
        # Finally: move to the save path
        with tempfile.TemporaryDirectory() as td:
            f_path = pathlib.Path(td, 'tmp.osm')
            g_path = pathlib.Path(td, 'tmp.gpkg')
            self.to_file(f_path, format='xml')
            gdal.VectorTranslate(str(g_path),
                                 str(f_path),
                                 dstSRS=srs,
                                 format='GPKG')
            ds = ogr.Open(str(g_path), update=True)
            ds.SetMetadata({
                'version': 0.6,
                'generator': self._generator,
                'timestamp_osm_base': self._timestamp,
                'copyright': COPYRIGHT_STR
            })
            ds = None
            g_path.rename(save_path)

        # Pop the added node IDs to save space
        for way in self._ways.values():
            way.tags.pop('osm_nodes', None)

    @classmethod
    def _from_json(cls, load_path: Union[str, pathlib.Path]) -> OSMNetwork:
        """ Import the network from a JSON file """
        # Get logger
        logger = get_logger()

        # Load JSON file
        with open(load_path, 'r') as f:
            d = json.load(f)

        # Constructor kwargs
        kwargs = {}

        # Read generator and timestamp
        if 'generator' in d:
            kwargs['generator'] = d['generator']
        osm3s = d.get('osm3s')
        if osm3s is not None and 'timestamp_osm_base' in osm3s:
            kwargs['timestamp'] = osm3s['timestamp_osm_base']

        # Read elements
        el = d['elements']     # fetch pointer for speed
        nodes = []     # type: List[OSMNode]
        ways = []     # type: List[OSMWay]
        for d in el:
            try:
                osm_element = OSMElementFactory.from_json_dict(d)
                if osm_element.__class__ == OSMNode:
                    nodes.append(osm_element)     # type: ignore
                elif osm_element.__class__ == OSMWay:
                    ways.append(osm_element)     # type: ignore
                else:
                    logger.warning(
                        f'Unhandled class {osm_element.__class__}! Skipping.')
                    continue
            except Exception as e:
                logger.warning(
                    'Exception raised trying to parse node! Skipping.')
                logger.exception(e, exc_info=True)
                continue

        # Create class!
        return cls.from_list(nodes, ways, **kwargs)

    @classmethod
    def _from_xml(cls, load_path: Union[str, pathlib.Path]) -> OSMNetwork:
        """ Import the network from an XML file """

        # Load XML file
        tree = ET.parse(str(load_path))
        root = tree.getroot()
        assert root.tag == 'osm'

        # Constructor kwargs
        kwargs = {}

        # Read generator and timestamp
        if 'generator' in root.attrib:
            kwargs['generator'] = root.attrib['generator']
        meta = root.find('meta')
        if meta is not None and 'osm_base' in meta.attrib:
            kwargs['timestamp'] = meta.attrib['osm_base']

        # Read elements
        nodes = []     # type: List[OSMNode]
        ways = []     # type: List[OSMWay]

        # Read nodes
        for node_xml in root.findall('node'):
            nodes.append(OSMNode.from_xml(node_xml))

        # Read ways
        for way_xml in root.findall('way'):
            ways.append(OSMWay.from_xml(way_xml))

        # Create class!
        return cls.from_list(nodes, ways, **kwargs)

    @staticmethod
    def _from_gpkg_fetch_tags(prop_d: Dict) -> Dict[str, str]:
        """ Fetch tags from an OGR JSON properties dict """
        # Output dict
        d = {}

        # Grab all non-null values in the top-level properties dict
        for k, v in prop_d.items():
            if v is not None:
                d[k] = v

        # Remove OSM ID: this is not a tag
        d.pop('osm_id', None)

        # Pop other_tags and parse directly
        # Fetch other_tags k -> v pairings and add to dict
        other_tags = d.pop('other_tags', None)
        if other_tags is not None:
            for tag_str in other_tags.split('","'):
                k, v = tag_str.split('"=>"')
                k = k.replace('"', '')
                v = v.replace('"', '')
                d[k] = v

        return d

    @classmethod
    @gdal_required(_gdal_available)
    def _from_gpkg(cls, load_path: Union[str, pathlib.Path]) -> OSMNetwork:
        """ Import the network from a GPKG file """
        # Get logger
        logger = get_logger()

        # Backwards counter for node IDs that are not
        # provided in the GPKG metadata
        osm_node_fake_id = -1

        # Node / way output dict
        nodes = {}
        ways = {}

        # Open dataset
        ds = ogr.Open(str(load_path))

        # Fetch top-level metadata
        kwargs = {}
        metadata = ds.GetMetadata()
        if metadata is not None:
            if 'generator' in metadata:
                kwargs['generator'] = metadata['generator']
            if 'generator' in metadata:
                kwargs['timestamp'] = metadata['timestamp_osm_base']

        # Get points layer, iterate and create all nodes
        # NOTE: Nodes here are (probably) not part of a way
        points_layer = ds.GetLayerByName('points')
        for _ in range(points_layer.GetFeatureCount()):
            feat = points_layer.GetNextFeature()
            feat_d = json.loads(feat.ExportToJson())

            lon, lat = feat_d['geometry']['coordinates']
            osm_id = int(feat_d['properties']['osm_id'])
            tags = cls._from_gpkg_fetch_tags(feat_d['properties'])

            node = OSMNode(id=osm_id, lat=lat, lon=lon, tags=tags)
            nodes[osm_id] = node

        # Get lines layer, iterate, and create all ways
        # and associated nodes
        lines_layer = ds.GetLayerByName('lines')
        for _ in range(lines_layer.GetFeatureCount()):
            feat = lines_layer.GetNextFeature()
            feat_d = json.loads(feat.ExportToJson())

            coords = feat_d['geometry']['coordinates']
            osm_id = int(feat_d['properties']['osm_id'])
            tags = cls._from_gpkg_fetch_tags(feat_d['properties'])

            # See if we can align nodes with their IDs
            osm_node_ids = tags.pop('osm_nodes', None)
            if osm_node_ids is not None:
                try:
                    osm_node_ids = json.loads(osm_node_ids)
                except json.JSONDecodeError:
                    logger.warning(
                        'Was not able to parse node IDs from osm_nodes! Got %s'
                        % repr(osm_node_ids))
                    osm_node_ids = None
                else:
                    if not len(coords) == len(osm_node_ids):
                        logger.warning(
                            'Coordinate and node ID lengths do not match! Cannot associated node IDs with coordinates.'
                        )
                        logger.debug(
                            f'Length Coords: {len(coords)}; Length OSM Node IDs: {len(osm_node_ids)}'
                        )
                        osm_node_ids = None

            # If we cannot associate nodes, iterate an integer counter backward for this way
            if osm_node_ids is None:
                logger.warning(
                    'Node IDs associated with ways not found or failed to parse (this metadata would not have been added by another source). Node IDs will be set to negative integers.'
                )
                osm_node_ids = []
                for _ in range(len(coords)):
                    osm_node_ids.append(osm_node_fake_id)
                    osm_node_fake_id -= 1

            # Finally we have a mapping of node IDs to coordinates (real or fake)
            # Create the associated nodes (if they do not exist)
            # Create and add this way with the referenced node IDs
            for node_id, (lon, lat) in zip(osm_node_ids, coords):
                if node_id not in nodes:
                    node = OSMNode(id=node_id, lat=lat, lon=lon)
                    nodes[node_id] = node

            way = OSMWay(id=osm_id, nodes=osm_node_ids, tags=tags)
            ways[osm_id] = way

        # Check other layers and let the user know it won't be parsed
        for layer_name in ('multilinestrings', 'multipolygons',
                           'other_relations'):
            layer = ds.GetLayerByName(layer_name)
            c = layer.GetFeatureCount()
            if c > 0:
                logger.warn(
                    '%d features found in layer \'%s\'. Features in this layer are not supported.'
                    % (c, layer_name))

        return cls.from_dict(nodes, ways, **kwargs)

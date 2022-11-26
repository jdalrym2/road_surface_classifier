#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from osgeo import ogr


class USGSDataType(ABC):

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    @abstractmethod
    def from_dict(cls, j: Dict[str, Any]):
        pass

    @classmethod
    def from_json_str(cls, s: str):
        return cls.from_dict(json.loads(s))


@dataclass(slots=True, frozen=True)
class AcquistionFilter(USGSDataType):
    start: datetime
    end: datetime

    def to_dict(self) -> Dict[str, Any]:
        return dict(start=self.start.isoformat(), end=self.start.isoformat())

    @classmethod
    def from_dict(cls, j: Dict[str, Any]):
        return cls(datetime.fromisoformat(j['start']),
                   datetime.fromisoformat(j['end']))


@dataclass(slots=True, frozen=True)
class Coordinate(USGSDataType):
    lat: float
    lon: float

    def to_dict(self) -> Dict[str, Any]:
        return dict(latitude=self.lat, longitude=self.lon)

    @classmethod
    def from_dict(cls, j: Dict[str, Any]):
        return cls(j['latitude'], j['longitude'])


@dataclass(slots=True, frozen=True)
class CloudCoverFilter(USGSDataType):
    min_pc: int
    max_pc: int
    include_unknown: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(min=self.min_pc,
                    max=self.max_pc,
                    includeUnknown=self.include_unknown)

    @classmethod
    def from_dict(cls, j: Dict[str, Any]):
        return cls(j['min'], j['max'], j['includeUnknown'])


@dataclass(slots=True, frozen=True)
class GeoJson(USGSDataType):
    geojson_type: str
    coordinates: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return dict(type=self.geojson_type, coordinates=self.coordinates)

    def to_ogr(self) -> ogr.Geometry:
        return ogr.CreateGeometryFromJson(json.dumps(self.to_dict()))

    @classmethod
    def from_ogr(cls, geom: ogr.Geometry):
        geojson = json.loads(geom.ExportToJson())
        return cls.from_dict(geojson)

    @classmethod
    def from_dict(cls, j: Dict[str, Any]):
        return cls(j['type'], j['coordinates'][0])


@dataclass(slots=True, frozen=True)
class SpatialFilterMbr(USGSDataType):
    lower_left: Coordinate
    upper_right: Coordinate

    def to_dict(self) -> Dict[str, Any]:
        return dict(filterType='mbr',
                    lowerLeft=self.lower_left.to_dict(),
                    upperRight=self.upper_right.to_dict())

    @classmethod
    def from_dict(cls, j: Dict[str, Any]):
        lower_left = Coordinate.from_dict(j['lowerLeft'])
        upper_right = Coordinate.from_dict(j['upperRight'])
        return cls(lower_left, upper_right)


@dataclass(slots=True, frozen=True)
class SpatialFilterGeoJson(USGSDataType):
    geoJson: GeoJson

    def to_dict(self) -> Dict[str, Any]:
        return dict(filterType='geoJson', geoJson=self.geoJson.to_dict())

    @classmethod
    def from_dict(cls, j: Dict[str, Any]):
        geoJson = GeoJson.from_dict(j['geoJson'])
        return cls(geoJson)


@dataclass(slots=True, frozen=True)
class SceneFilter(USGSDataType):
    acquistion_filter: Optional[AcquistionFilter] = None
    cloud_cover_filter: Optional[CloudCoverFilter] = None
    dataset_name: Optional[str] = None
    ingest_filter: Optional[None] = None     # TODO: if needed
    metadata_filter: Optional[None] = None     # TODO: if needed
    seasonal_filter: Optional[List[int]] = None
    spatial_filter: Optional[Union[SpatialFilterMbr,
                                   SpatialFilterGeoJson]] = None

    def to_dict(self) -> Dict[str, Any]:
        out_dict = dict()
        if self.acquistion_filter is not None:
            out_dict['acquistionFilter'] = self.acquistion_filter.to_dict()
        if self.cloud_cover_filter is not None:
            out_dict['cloudCoverFilter'] = self.cloud_cover_filter.to_dict()
        if self.dataset_name is not None:
            out_dict['datasetName'] = self.dataset_name
        if self.ingest_filter is not None:
            pass     # TODO: if needed
        if self.metadata_filter is not None:
            pass     # TODO: if needed
        if self.seasonal_filter is not None:
            out_dict['seasonalFilter'] = self.seasonal_filter
        if self.spatial_filter is not None:
            out_dict['spatialFilter'] = self.spatial_filter.to_dict()

        return out_dict

    @classmethod
    def from_dict(cls, j: Dict[str, Any]):
        return cls(j['datasetName'], j.get('acquistionFilter'),
                   j.get('cloudCoverFilter'), j.get('ingestFilter'),
                   j.get('metadataFilter'), j.get('seasonalFilter'),
                   j.get('spatialFilter'))

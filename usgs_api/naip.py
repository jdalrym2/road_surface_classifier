#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List

from osgeo import ogr, osr

from .scene_search import SceneSearchRequest

# {'browse': [{'id': '5e83a3406798f5c8',
#    'browseRotationEnabled': None,
#    'browseName': 'Standard Browse',
#    'browsePath': 'https://ims.cr.usgs.gov/browse/naip/fullres/CO/2019/201908_colorado_naip_0x6000m_utm_cnir/39104/m_3910464_sw_13_060_20190919.jpg',
#    'overlayPath': 'https://ims.cr.usgs.gov/browse/naip/fullres/CO/2019/201908_colorado_naip_0x6000m_utm_cnir/39104/m_3910464_sw_13_060_20190919.jpg',
#    'overlayType': 'file',
#    'thumbnailPath': 'https://ims.cr.usgs.gov/thumbnail/naip/fullres/CO/2019/201908_colorado_naip_0x6000m_utm_cnir/39104/m_3910464_sw_13_060_20190919.jpg'}],
#  'cloudCover': None,
#  'entityId': '3031389',
#  'displayId': 'M_3910464_SW_13_060_20190919',
#  'orderingId': None,
#  'metadata': [],
#  'hasCustomizedMetadata': None,
#  'options': {'bulk': True,
#   'download': True,
#   'order': False,
#   'secondary': False},
#  'selected': {'bulk': False, 'compare': False, 'order': False},
#  'spatialBounds': {'type': 'Polygon',
#   'coordinates': [[[-104.1280971, 38.9976888],
#     [-104.1280971, 39.0648194],
#     [-104.0593222, 39.0648194],
#     [-104.0593222, 38.9976888],
#     [-104.1280971, 38.9976888]]]},
#  'spatialCoverage': {'type': 'Polygon',
#   'coordinates': [[[-104.1280971, 38.9982138],
#     [-104.0602027, 38.9976888],
#     [-104.0593222, 39.0642888],
#     [-104.1272804, 39.0648194],
#     [-104.1280971, 38.9982138]]]},
#  'temporalCoverage': {'endDate': '2019-09-19 00:00:00-05',
#   'startDate': '2019-09-19 00:00:00-05'},
#  'publishDate': '2021-03-18 12:22:01.205175-05'}


class NAIPSceneSearchResult():

    __slots__ = [
        'overlay_url', 'entity_id', 'display_id', 'spatial_coverage_json',
        'acq_time'
    ]

    def __init__(self, overlay_url: str, entity_id: str, display_id: str,
                 spatial_coverage_json: Dict[str, Any], acq_time: datetime):
        self.overlay_url = overlay_url
        self.entity_id = entity_id
        self.display_id = display_id
        self.spatial_coverage_json = spatial_coverage_json
        self.acq_time = acq_time

    def spatial_coverage(self) -> ogr.Geometry:
        return ogr.CreateGeometryFromJson(
            json.dumps(self.spatial_coverage_json))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        overlay_url = d.get('browse', [{}])[0].get('overlayPath', '')
        entity_id = d['entityId']
        display_id = d['displayId']
        spatial_coverage_json = d['spatialCoverage']
        acq_time_str = d['temporalCoverage']['startDate'].split(' ')[0]
        acq_time = datetime.strptime(acq_time_str, r'%Y-%m-%d')
        return cls(overlay_url, entity_id, display_id, spatial_coverage_json,
                   acq_time)


@dataclass(kw_only=True, slots=True, frozen=True)
class NAIPSceneSearchRequest(SceneSearchRequest):
    dataset_name: str = 'naip'


def naip_results_to_shapefile(results: List[NAIPSceneSearchResult]):

    # Create SRS (EPSG:4326: WGS-84 decimal degrees)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource('/data/gis/jon_tmp.shp')
    layer = ds.CreateLayer('results', srs=srs, geom_type=ogr.wkbPolygon)

    acq_date_field = ogr.FieldDefn('acq_date', ogr.OFTDate)
    layer.CreateField(acq_date_field)
    feature_defn = layer.GetLayerDefn()

    for result in results:
        feat = ogr.Feature(feature_defn)
        feat.SetGeometry(result.spatial_coverage())
        feat.SetField('acq_date', datetime.strftime(result.acq_time,
                                                    '%Y-%m-%d'))
        layer.CreateFeature(feat)
        feat = None

    layer = None
    ds = None

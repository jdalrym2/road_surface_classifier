#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import USGS_API_URL
from .data_types import SceneFilter
from .request import USGSAPIRequest


@dataclass(kw_only=True, slots=True, frozen=True)
class SceneSearchRequest(USGSAPIRequest):
    dataset_name: str
    endpoint: str = USGS_API_URL + 'scene-search'
    starting_number: int = 0
    max_results: int = 100
    scene_filter: Optional[SceneFilter] = None

    def to_dict(self) -> Dict[str, Any]:
        out_dict: Dict[str, Any] = dict(datasetName=self.dataset_name,
                                        startingNumber=self.starting_number,
                                        maxResults=self.max_results)
        if self.scene_filter is not None:
            out_dict['sceneFilter'] = self.scene_filter.to_dict()

        return out_dict

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import json
from typing import Any, Optional, Dict
from dataclasses import dataclass
import requests

from .response import USGSAPIResponse
from . import get_api_key


@dataclass(slots=True, frozen=True)
class USGSAPIRequest(ABC):
    endpoint: str

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_response(self) -> requests.Response:
        return requests.post(self.endpoint,
                             headers={'X-Auth-Token': get_api_key()},
                             json=self.to_dict())

    def make_request(self) -> 'USGSAPIResponse':
        return USGSAPIResponse.from_response(self.to_response())

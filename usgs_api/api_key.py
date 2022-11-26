#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict
import requests

from . import USGS_API_URL, MODULE_CONFIG, get_logger

from .response import USGSAPIResponse


@dataclass(kw_only=True, slots=True, frozen=True)
class APIKeyRequest():
    username: str
    password: str
    endpoint: str = USGS_API_URL + 'login'

    def make_request(self) -> 'USGSAPIResponse':
        return USGSAPIResponse.from_response(self.to_response())

    def to_dict(self) -> Dict[str, Any]:
        return dict(username=self.username, password=self.password)

    def to_response(self) -> requests.Response:
        # Since this is the API key request, we do not include the API key in the headers
        return requests.post(self.endpoint, headers={}, json=self.to_dict())

    @classmethod
    def from_module_config(cls):
        return cls(username=MODULE_CONFIG.username,
                   password=MODULE_CONFIG.password)


def _get_api_key() -> str:
    """ Fetch an API key from the USGS API endpoint """

    get_logger().info('Fetching API key...')

    # HTTP post request to the login endpoint w/ username and password
    response = APIKeyRequest.from_module_config().make_request()

    return response.data
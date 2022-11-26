#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests

from . import USGS_API_URL, MODULE_CONFIG, get_logger

from .usgs_api_response import USGSAPIResponse


def _get_api_key() -> str:
    """ Fetch an API key from the USGS API endpoint """
    LOGIN_ENDPOINT = USGS_API_URL + 'login'

    get_logger().info('Fetching API key...')

    # HTTP post request to the login endpoint w/ username and password
    response = requests.post(LOGIN_ENDPOINT,
                             json=dict(username=MODULE_CONFIG.username,
                                       password=MODULE_CONFIG.password))

    # Load response (throws error if not response.ok or if the API response has an error)
    usgs_response = USGSAPIResponse.from_response(response)

    return usgs_response.data
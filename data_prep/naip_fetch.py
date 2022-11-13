#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests

USGS_API_URL = r'https://m2m.cr.usgs.gov/api/api/json/stable/'

# TODO: don't hardcode, obvs
username = r''
password = r''
api_key = r''


class USGSAPIResponse():
    pass


def get_api_key() -> str:
    """ Fetch an API key from the USGS API endpoint """
    LOGIN_ENDPOINT = USGS_API_URL + 'login'

    # HTTP post request to the login endpoint w/ username and password
    response = requests.post(LOGIN_ENDPOINT,
                             json=dict(username=username, password=password))

    # Check response status code
    if response.ok:
        # Parse response JSON data
        response_data = response.json()
        # If no error code, return API key
        if response_data.get('errorCode') is None:
            return str(response_data['data'])
        else:
            raise RuntimeError(
                f'Got error code {response_data.get("errorCode")}. Msg: {response_data.get("errorMessage")}'
            )
    else:
        raise RuntimeError('API Key Request Reponse returned status code: %d' %
                           response.status_code)


if __name__ == '__main__':
    get_api_key()
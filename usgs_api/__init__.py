#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import pathlib
import logging
from typing import Any, Dict, Optional

USGS_API_URL = r'https://m2m.cr.usgs.gov/api/api/json/stable/'

# Load config schema
_CONFIG_SCHEMA_LOC = pathlib.Path(
    __file__).parent / 'data' / 'config_schema.json'
if not _CONFIG_SCHEMA_LOC.exists():
    raise RuntimeError(
        'Could not find config schema at %s! Cannot init module.' %
        str(_CONFIG_SCHEMA_LOC))
with open(_CONFIG_SCHEMA_LOC, 'r') as f:
    _CONFIG_SCHEMA: Dict[str, Any] = json.load(f)

# Python module config location
_CONFIG_LOC = pathlib.Path().home() / '.config' / '.usgs_api'
_CONFIG_LOC.parent.mkdir(exist_ok=True)

# Other module state variables
_api_key: Optional[str] = None

# Configure logger
_logger = logging.getLogger(__name__)
_logger.setLevel(
    logging.DEBUG)     # configure log level here. TODO: add to ~/.usgs_api
_logger_handlers = [logging.StreamHandler(sys.stdout)]
_logger_formatter = logging.Formatter(
    r'%(asctime)-15s %(levelname)s [%(module)s] %(message)s')
_logger.handlers.clear()
for h in _logger_handlers:
    h.setFormatter(_logger_formatter)
    _logger.addHandler(h)

# Load from config
from .config import _load_config, _set_config

if _CONFIG_LOC.exists():
    success = True
    try:
        MODULE_CONFIG = _load_config(_CONFIG_LOC, _CONFIG_SCHEMA)
    except Exception as e:
        _logger.exception(e)
        _logger.warning('Failed to load config from file!')
        success = False

    # If we didn't read the config file correctly, attempt an unlink to
    # refresh the state
    if not success:
        _logger.info('Attempting to unlink existing config file.')
        try:
            _CONFIG_LOC.unlink()
        except Exception as e:
            _logger.exception(e)
            _logger.error('Unlink failed! Continuing anyway.')
        MODULE_CONFIG = _set_config(_CONFIG_LOC, _CONFIG_SCHEMA)
    del success
else:
    MODULE_CONFIG = _set_config(_CONFIG_LOC, _CONFIG_SCHEMA)


def get_logger() -> logging.Logger:
    """ Fetch the usgs_api logger """
    return _logger


from .api_key import _get_api_key


def get_api_key(reset: bool = False) -> str:
    """ Fetch an API key from the USGS API endpoint """
    global _api_key
    # TODO: API key timeout
    if reset or _api_key is None:
        _api_key = _get_api_key()
    return _api_key
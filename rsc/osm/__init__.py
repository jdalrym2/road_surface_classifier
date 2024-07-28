#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import functools

# Configure logger
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)     # configure log level here
_logger_handlers = [logging.StreamHandler(sys.stdout)]
_logger_formatter = logging.Formatter(
    r'%(asctime)-15s %(levelname)s [%(module)s] %(message)s')
_logger.handlers.clear()
for h in _logger_handlers:
    h.setFormatter(_logger_formatter)
    _logger.addHandler(h)

def get_logger():
    """ Fetch the module logger """
    return _logger

def gdal_required(is_available):
    """ Decorator function to check if GDAL was imported """
    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_available:
                raise RuntimeError('OGR is required for this function / method.')
            return func(*args, **kwargs)
        return wrapper
    return _decorator

# Exposed classes to user
from .osm_element import OSMElement, OSMNode, OSMWay
from .osm_element_factory import OSMElementFactory
from .osm_network import OSMNetwork
from . import overpass_api
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64
import json
import getpass
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional

import jsonschema

from . import _logger


@dataclass(slots=True, kw_only=True, frozen=True)
class ModuleConfig:
    """
    Holds configuration values for this module
    """
    username: str
    b64password: bytes
    download_dir: pathlib.Path

    @property
    def password(self):
        return base64.decodebytes(self.b64password).decode('utf-8')


def _load_config(config_loc: pathlib.Path,
                 config_schema: Dict[str, str]) -> ModuleConfig:

    if not config_loc.exists():
        _logger.info('Could not find config file!')
        raise ValueError('Could not find config file!')

    # Load config
    try:
        with open(config_loc) as f:
            config: Dict[str, str] = json.load(f)
            jsonschema.validate(config, config_schema,
                                jsonschema.Draft7Validator)
    except Exception as e:
        _logger.exception(e)
        _logger.error("Could not load and verify config file. Unlinking.")
        raise e

    # Read username and password
    username = config['username']
    b64password = config['b64password'].encode('utf-8')

    # Read download dir
    conf_download_dir = pathlib.Path(config['download_dir'])
    if conf_download_dir.exists():
        download_dir = conf_download_dir
    else:
        _logger.error('Configured download directory does not exist!')
        raise RuntimeError('Configured download directory does not exist!')

    return ModuleConfig(username=username,
                        b64password=b64password,
                        download_dir=download_dir)


def _set_config(config_loc: pathlib.Path,
                config_schema: Dict[str, str]) -> ModuleConfig:

    # Have the user input username and password info
    username: str = input('Input USGS M2M API username: ')
    b64password: bytes = base64.encodebytes(
        getpass.getpass('Input USGS M2M API password: ').strip().encode(
            'utf-8'))
    download_dir: Optional[pathlib.Path] = None
    while download_dir is None or not download_dir.is_dir():
        download_dir = pathlib.Path(
            input(
                'Input on-disk download directory for USGS API data (must be an existing directory): '
            )).resolve()
        if not download_dir.is_dir():
            _logger.warning('Input is not a directory! %s' % str(download_dir))

    # Ask the user whether or not to save the configuration
    save_config = ''
    while save_config.lower() not in ('y', 'n'):
        save_config = input('Save to %s? (y/n): ' % config_loc)

    # If so, save!
    if save_config.lower() == 'y':
        config = {
            'username': username,
            'b64password': b64password.decode('utf-8'),
            'download_dir': str(download_dir)
        }
        jsonschema.validate(config, config_schema, jsonschema.Draft7Validator)
        with open(config_loc, 'w') as f:
            json.dump(config, f, indent=2)
        return _load_config(config_loc, config_schema)

    else:
        return ModuleConfig(username=username,
                            b64password=b64password,
                            download_dir=download_dir)

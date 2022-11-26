#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Optional, Dict
from dataclasses import dataclass
import requests


@dataclass(kw_only=True, slots=True)
class USGSAPIResponse():
    request_id: int
    version: str
    data: Any
    error_code: Optional[int]
    error_message: Optional[str]
    raise_on_error: bool = True

    def __post_init__(self):
        if self.raise_on_error:
            self.raise_exc()

    def is_error(self):
        return self.error_code is not None

    def raise_exc(self):
        if self.is_error():
            raise RuntimeError('[%d]: %s' % self.error_code,
                               self.error_message)

    @classmethod
    def from_response(cls, response: requests.Response) -> 'USGSAPIResponse':
        if response.ok:
            return cls.from_json(response.json())
        raise ValueError('API returned HTTP response code: %d' %
                         response.status_code)

    @classmethod
    def from_json(cls, j: Dict[str, Any]) -> 'USGSAPIResponse':
        return cls(request_id=j['requestId'],
                   version=j['version'],
                   data=j['data'],
                   error_code=j['errorCode'],
                   error_message=j['errorMessage'])
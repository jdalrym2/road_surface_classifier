#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import traceback
from abc import ABC, abstractmethod
from typing import Optional

from torch.utils.data import DataLoader

from . import device


class MetricsHandler(ABC):

    def __init__(self, output_dir: pathlib.Path, model,
                 dataloader: DataLoader):
        self.output_dir = output_dir
        self.model = model
        self.model.to(device)
        self.dataloader = dataloader

    def __call__(self, raise_on_error: bool = False) -> Optional[pathlib.Path]:
        try:
            return self.generate_artifact()
        except Exception:
            if raise_on_error:
                raise
            traceback.print_exc()
            print('Exception raised! Skipping artifact generation.')

    @abstractmethod
    def generate_artifact(self) -> pathlib.Path:
        pass

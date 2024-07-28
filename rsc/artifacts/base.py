#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import traceback
from abc import ABC, abstractmethod
from typing import Any, Sequence

from tqdm import tqdm

from torch.utils.data import DataLoader

from . import device


class ArtifactHandler(ABC):
    """ Abstract class used to generate an artifact from an inference result """

    def __init__(self):
        pass

    @abstractmethod
    def start(self, model: Any, dataloader: DataLoader) -> None:
        """
        Handler startup code. Do any initalization here.

        Args:
            model (Any): Model that will be used
            dataloader (DataLoader): Dataloader that will be used
        """
        pass

    @abstractmethod
    def on_iter(self, dl_iter: Sequence, model_out: Sequence) -> None:
        """
        Called each time a batch is inferenced. Collect data for
        the artifact here.

        Args:
            dl_iter (Sequence): Output iterable from the dataloader.
            model_out (Sequence): Output iterable from `model.predict`
        """
        pass

    @abstractmethod
    def save(self, output_dir: pathlib.Path) -> pathlib.Path:
        """
        Given the collected state from on_iter, generate and
        save the relevant artifact.

        Args:
            output_dir (pathlib.Path): Output directory to artifact

        Returns:
            pathlib.Path: Path to saved artifact
        """
        pass


class ArtifactGenerator:
    """ Top-level class to handle multiple artifact generation """

    def __init__(self, output_dir: str | pathlib.Path, model: Any,
                 dataloader: DataLoader):
        """
        Create an artifact generator

        Args:
            output_dir (str | pathlib.Path): Output directory to dump artifacts
            model (Any): RSC model for inference
            dataloader (DataLoader): Dataloader to run inference over
        """
        self.handlers: list[ArtifactHandler] = []
        self.is_active: list[bool] = []
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=False, exist_ok=True)
        self.model = model
        self.model.to(device)
        self.dataloader = dataloader

    def add_handler(self, handler: ArtifactHandler):
        """
        Add a handler to the generator. All handlers should be added
        before `run` is called.

        Args:
            handler (ArtifactHandler): Handler to add to the generator.

        Raises:
            ValueError: If the handler is not a subclass of `ArtifactHandler`
        """
        if not issubclass(handler.__class__, ArtifactHandler):
            raise ValueError(
                f'Input handler must be a subclass of {ArtifactHandler.__name__}'
            )
        self.handlers.append(handler)

    def _exec(self, idx, func, raise_on_error: bool) -> Any:
        """ Wrapper call to work with the handler. Provides
            a failsafe in the case the handler throws an exception
            such that inference can continue """
        # Skip an inactive handler
        if not self.is_active[idx]:
            return
        try:
            # Attempt call
            return func()
        except Exception:
            # If we failed, raise if we want
            if raise_on_error:
                raise
            # Otherwise, print the exception and deactivate
            # the handler
            traceback.print_exc()
            print(f'Deactivating handler {idx:d}...')
            self.is_active[idx] = False

    def run(self, raise_on_error: bool = False):
        """
        Run inference and pass data to each handler.

        Args:
            raise_on_error (bool, optional): If any handler throws an error,
                whether or not to raise. Defaults to False.
        """

        # Ready?
        self.is_active = [True for _ in self.handlers]

        # Set...
        for idx, h in enumerate(self.handlers):
            self._exec(idx, lambda: h.start(self.model, self.dataloader),
                       raise_on_error)

        # Go!
        for dl_iter in tqdm(iter(self.dataloader)):
            x, _ = dl_iter

            # Extract just the image + location mask for inference
            # i.e. strip the last (probmask) channel
            x = x[:, :-1, :, :].to(device)

            # Get prediction from model
            model_out = self.model(x)

            # Run the handlers
            for idx, h in enumerate(self.handlers):
                self._exec(idx, lambda: h.on_iter(dl_iter, model_out),
                           raise_on_error)

        # Finalize
        for idx, h in enumerate(self.handlers):
            self._exec(idx, lambda: h.save(self.output_dir), raise_on_error)

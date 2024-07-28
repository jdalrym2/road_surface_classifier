#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Generate artifacts for a model """

import pathlib
import argparse

import torch
from torch.utils.data import DataLoader

from ..train.plmcnn import PLMaskCNN
from ..train.dataset import RoadSurfaceDataset
from ..train.preprocess import PreProcess

from . import find_best_model
from .base import ArtifactGenerator
from .confusion_matrix_handler import ConfusionMatrixHandler
from .accuracy_obsc_handler import AccuracyObscHandler
from .obsc_compare_handler import ObscCompareHandler
from .samples_handler import SamplesHandler
from .auc_handler import AUCHandler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i',
                        '--input-ckpt',
                        type=str,
                        required=True,
                        help='Path to input checkpoint file. '
                        'If given a directory, attempts to '
                        'find the best model in the directory.')
    parser.add_argument('-d',
                        '--dataset-csv',
                        type=str,
                        required=True,
                        help='Path to dataset CSV')
    parser.add_argument('-o',
                        '--output-path',
                        type=str,
                        required=True,
                        help='Path to output directory for artifacts. '
                        'If it does not exist, one will be created.')
    parser.add_argument('-m',
                        '--model-path',
                        type=str,
                        required=False,
                        help='Path to model PTH file, '
                        'if PLMaskCNN is not the model')
    parser.add_argument('-c',
                        '--count',
                        type=int,
                        required=False,
                        default=-1,
                        help='Maximum of images to do '
                        'inference on. '
                        '-1: inference all images')
    parser.add_argument('--batch-size',
                        type=int,
                        required=False,
                        default=64,
                        help='Set dataloader batch size.')
    parser.add_argument('--num-workers',
                        type=int,
                        required=False,
                        default=16,
                        help='Set dataloader worker count.')
    parser.add_argument('--no-shuffle',
                        action='store_true',
                        help='If specified, do not shuffle '
                        'the dataset when loading.')
    parser.add_argument('--raise-on-error',
                        action='store_true',
                        help='If specified, stop processing when '
                        'encountering an error.')

    return parser.parse_args()


if __name__ == '__main__':

    pargs = parse_args()

    # Parse relevant inputs up front
    csv_path = pathlib.Path(pargs.dataset_csv)
    assert csv_path.is_file()

    # Determine model checkpoint
    ckpt_path = pathlib.Path(pargs.input_ckpt)
    if ckpt_path.is_file():
        # Load the checkpoint as-is
        pass
    elif ckpt_path.is_dir():
        # Find the best model from the directory
        ckpt_path = find_best_model(ckpt_path)
        print('Found best model: %s' % str(ckpt_path))
    else:
        raise ValueError(
            'Could not find checkpoint path: %s. Must be a file or directory.'
            % str(ckpt_path))

    # Load model
    if pargs.model_path is None:
        model = PLMaskCNN.load_from_checkpoint(ckpt_path)
    else:
        model_path = pathlib.Path(pargs.model_path)
        assert model_path.is_file()
        model: PLMaskCNN = torch.load(model_path)
        model.load_from_checkpoint(ckpt_path)

    # Put model in eval mode
    model.eval()

    # Construct dataset
    val_ds = RoadSurfaceDataset(csv_path,
                                transform=PreProcess(),
                                n_channels=model.nc,
                                limit=pargs.count)

    # Construct dataloader
    val_dl = DataLoader(val_ds,
                        num_workers=pargs.num_workers,
                        batch_size=pargs.batch_size,
                        shuffle=not pargs.no_shuffle)

    # Parse save directory, create if not there
    save_dir = pathlib.Path(pargs.output_path)
    if not save_dir.is_dir():
        save_dir.mkdir(parents=False)

    # Generate artifacts from model
    generator = ArtifactGenerator(save_dir, model, val_dl)
    generator.add_handler(ConfusionMatrixHandler(simple=True))
    generator.add_handler(ConfusionMatrixHandler(simple=False))
    generator.add_handler(AccuracyObscHandler())
    generator.add_handler(ObscCompareHandler())
    generator.add_handler(SamplesHandler())
    # generator.add_handler(AUCHandler())  # disabled for multiclass
    generator.run(raise_on_error=pargs.raise_on_error)

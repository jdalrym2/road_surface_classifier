#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" RSC submodule to simplify artifact generation during inference over a dataset """
import pathlib

import torch

# Get PyTorch device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_best_model(results_dir: pathlib.Path) -> pathlib.Path:
    """
    Find the best model checkpoint (by validation loss) in a results directory.

    Args:
        results_dir (pathlib.Path): Results directory

    Returns:
        pathlib.Path: Path to checkpoint with minimum validation loss.
    """
    # Parse checkpoints from results dir
    paths = list(results_dir.glob('*.ckpt'))
    path_stems = [e.stem for e in paths]
    path_metrics = [e.split('-') for e in path_stems]
    # Exact validation losses from filenames
    val_losses = []
    for idx, metrics in enumerate(path_metrics):
        for metric in metrics:
            if metric.startswith('val_loss'):
                val_loss = float(metric.split('=')[-1])
                val_losses.append((idx, val_loss))

    # Find path that has minimum validation loss
    # By reversing val losses, if there is a tie
    # we fetch the last one
    min_idx, _ = min(reversed(val_losses), key=lambda v: v[1])

    return paths[min_idx]


from .base import ArtifactGenerator, ArtifactHandler
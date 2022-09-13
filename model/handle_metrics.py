#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Metrics handling code """
import pathlib
import traceback
from typing import List, Optional
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set matplotlib backend
# matplotlib.use('Agg')

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
    min_idx, _ = min(val_losses, key=lambda v: v[1])

    return paths[min_idx]


def generate_plots(results_dir: pathlib.Path,
                   val_dl: Optional[DataLoader] = None) -> None:
    """
    Generate series of plots given a results directory and validation data

    Args:
        results_dir (pathlib.Path): Results directory
        val_dl (Optional[DataLoader], optional): Validation dataloader,
            if not given, relevant plots will not be generated. Defaults to None.
    """
    global device

    # Look for metrics.csv
    csv_path = results_dir / 'lightning_logs' / 'version_0' / 'metrics.csv'
    csv_path_found = csv_path.is_file()
    if not csv_path_found:
        warnings.warn(
            RuntimeWarning('Could not find metrics.csv! Looked here: %s' %
                           str(csv_path)))

    # Look for model.pth
    model_path = model_path = results_dir / 'model.pth'
    model_path_found = model_path.is_file()
    if not model_path_found:
        warnings.warn(
            RuntimeWarning('Could not find model.pth! Looked here: %s' %
                           str(model_path)))

    # Look for best model
    best_model_found = False
    best_model = pathlib.Path()
    try:
        best_model_found = True
        best_model = find_best_model(results_dir)
        print('Found best model: %s' % str(best_model))
    except Exception:
        print('Exception occurred looking for best model:')
        traceback.print_exc()

    plot_sample(model_path, best_model, val_dl)
    return

    # Plot metrics
    if csv_path_found:
        plot_metrics(csv_path, results_dir / 'metrics_plot.png')
    else:
        print('Not generating metrics plot...')

    # Plot confusion matrix
    if model_path_found and best_model_found and val_dl is not None:
        plot_confusion_matrix(model_path=model_path,
                              ckpt_path=best_model,
                              val_dl=val_dl,
                              output_path=results_dir / 'cm.png')
    else:
        print('Not generating confusion matrix')


def plot_metrics(csv_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """
    Generate plots from metrics.csv file.

    Args:
        csv_path (pathlib.Path): Path to metrics.csv file to generate plots from
        output_path (pathlib.Path): Output plot
    """
    # CSVLogger will have multiple rows per epoch, with only
    # 1 non-NaN metric per row. This will collapse all the metrics
    # together to one row per epoch
    df = pd.read_csv(csv_path).groupby('epoch').mean()

    # Drop the step column so we don't plot it
    df = df.drop(columns=[
        'step', 'train_loss1', 'train_loss2', 'val_loss1', 'val_loss2'
    ])

    # Generate plots!
    fig, ax = plt.subplots()
    df.plot(ax=ax, subplots=True, backend='matplotlib')
    plt.savefig(str(output_path))
    plt.close(fig)


def plot_confusion_matrix(model_path: pathlib.Path,
                          ckpt_path: pathlib.Path,
                          val_dl: DataLoader,
                          output_path: pathlib.Path,
                          labels: Optional[List[str]] = None):
    """
    Plot a confusion matrix for a model.

    Args:
        model_path (pathlib.Path): Path to model to load
        ckpt_path (pathlib.Path): Path to model checkpoint to load
        val_dl (DataLoader): Path to validation DataLoader to use
        output_path (pathlib.Path): Plot output path
        labels (Optional[List[str]], optional): Class labels. If not provided,
            will be attempted to be extracted from the model. Otherwise, non-specific
            values will be used. Defaults to None.
    """
    # Load model and checkpoint

    print(model_path)
    print(ckpt_path)

    model = torch.load(model_path).load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # Try to get labels
    if labels is None:
        labels = model.__dict__.get('labels')

    # Get truth and predictions using the dataloader
    y_true_l, y_pred_l = [], []
    for x, features in tqdm(iter(val_dl)):
        xm = x[:, 3:, :, :]
        x = x[:, :3, :, :]
        y_true_l.append(features.numpy())
        # predict with the model
        y_pred = torch.argmax(model(x.to(device), xm.to(device))[1],
                              axis=1).cuda()     # type: ignore
        y_pred_l.append(y_pred.cpu().numpy())
    y_true = np.concatenate(y_true_l)
    y_pred = np.concatenate(y_pred_l)

    # Generate and save the confusion matrix
    c = ConfusionMatrixDisplay(confusion_matrix(y_true,
                                                y_pred,
                                                normalize='true'),
                               display_labels=labels)
    fig, ax = plt.subplots()
    c.plot(ax=ax, cmap=plt.cm.Blues)     # type: ignore
    fig.savefig(str(output_path))
    plt.close(fig)


def plot_sample(model_path: pathlib.Path, ckpt_path: pathlib.Path,
                val_dl: DataLoader):

    # Load model and checkpoint
    print(model_path)
    print(ckpt_path)

    model = torch.load(model_path).load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # Get truth and predictions using the dataloader
    y_true_l, y_pred_l = [], []
    for x, features in tqdm(iter(val_dl)):
        xm = x[:, 3:, :, :]
        x = x[:, :3, :, :]
        y_true_l.append(features.numpy())
        # predict with the model

        ym, y_pred = model(x.to(device), xm.to(device))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
        for i in range(4):
            ax[i, 0].imshow(xm[i, 0, ...].cpu().detach(), vmin=-1, vmax=1)
            ax[i, 1].imshow(ym[i, 0, ...].cpu().detach(), vmin=-1, vmax=1)
        plt.show()

        break


if __name__ == '__main__':

    results_dir = pathlib.Path(
        '/home/jon/git/road_surface_detection/results/20220911_024715Z')
    assert results_dir.is_dir()

    # Import dataset
    from custom_image_dataset import RoadSurfaceDataset
    from data_augmentation import PreProcess
    preprocess = PreProcess()
    val_ds = RoadSurfaceDataset(
        '/data/road_surface_detection/dataset_simple/dataset_val.csv',
        transform=preprocess)
    val_dl = DataLoader(val_ds, num_workers=16, batch_size=16, shuffle=False)

    generate_plots(results_dir, val_dl)

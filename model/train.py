#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Set AWS profile (for use in MLFlow)
import os

os.environ["AWS_PROFILE"] = 'truenas'
os.environ["MLFLOW_S3_ENDPOINT_URL"] = 'http://truenas.local:9807'

import pathlib
from typing import List, Type
from datetime import datetime

import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from plmcnn import PLMaskCNN
from preprocess import PreProcess
from road_surface_dataset import RoadSurfaceDataset
from generate_artifacts.metrics_handler import MetricsHandler
from generate_artifacts.confusion_matrix_handler import ConfusionMatrixHandler

if __name__ == '__main__':

    # Create directory for results
    results_dir = pathlib.Path(
        '/data/road_surface_classifier/results').resolve()
    assert results_dir.is_dir()
    save_dir = results_dir / datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')
    save_dir.mkdir(parents=False, exist_ok=False)

    # Save checkpoints (model states for later)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        monitor='val_loss',
        save_top_k=3,
        filename='model-{epoch:02d}-{val_loss:.5f}')

    # Setup early stopping based on validation loss
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            mode='min',
                                            patience=10)

    # Log epoch + validation loss to CSV
    #logger = CSVLogger(str(save_dir))
    mlflow_logger = MLFlowLogger(experiment_name='road_surface_classifier',
                                 run_name='run_%s' %
                                 datetime.utcnow().strftime('%Y%m%d_%H%M%SZ'),
                                 tracking_uri='http://truenas.local:9809')

    # Get dataset
    preprocess = PreProcess()
    train_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset/dataset_train.csv',
        transform=preprocess,
        limit=500)
    val_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset/dataset_val.csv',
        transform=preprocess,
        limit=500)

    # Create data loaders.
    batch_size = 64
    train_dl = DataLoader(train_ds,
                          num_workers=16,
                          batch_size=batch_size,
                          shuffle=True)
    val_dl = DataLoader(val_ds, num_workers=16, batch_size=batch_size)

    # Labels and weights
    # Load class weights
    weights_df = pd.read_csv(
        '/data/road_surface_classifier/dataset/class_weights.csv')

    # NOTE: We add obscuration class with a weight of 1
    labels = list(weights_df['class_name']) + ['Obscured']
    weights = list(weights_df['weight']) + [1]

    # Model
    model = PLMaskCNN(labels=labels, weights=weights)
    torch.save(model, save_dir / 'model.pth')

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=mlflow_logger)

    # Do the thing!
    trainer.fit(model, train_dataloaders=train_dl,
                val_dataloaders=val_dl)     # type: ignore

    # Upload best model
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id,
                                          checkpoint_callback.best_model_path)

    # Plot confusion matrix
    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.eval()

    # Generate artifacts from model
    metrics_handlers: List[Type[MetricsHandler]] = [ConfusionMatrixHandler]
    for handler_class in metrics_handlers:

        # Generate artifact
        artifact_path = handler_class(save_dir, model, val_dl)()

        # Upload artifact
        if artifact_path is not None:
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id,
                                                  str(artifact_path))
        else:
            print('Skipping artifact upload.')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from typing import Optional

sys.path.append(
    os.getcwd())     # TODO: this is silly, would be fixed with pip install

# Set AWS profile (for use in MLFlow)
os.environ["AWS_PROFILE"] = 'truenas'
os.environ["MLFLOW_S3_ENDPOINT_URL"] = 'http://truenas.local:9807'

import pathlib
from datetime import datetime

import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import MLFlowLogger

from rsc.train.plmcnn import PLMaskCNN
from rsc.train.preprocess import PreProcess
from rsc.train.dataset import RoadSurfaceDataset
from rsc.artifacts.confusion_matrix_handler import ConfusionMatrixHandler

# Recommended for CUDA
torch.set_float32_matmul_precision('medium')

# Whether or not to include the NIR channel
# since the input dataset includes it
INCLUDE_NIR = True

# Quick test, train an epic on fewer-than-normal
# set of images to check everything works
QUICK_TEST = False

if __name__ == '__main__':

    # Create directory for results
    results_dir = pathlib.Path(
        '/data/road_surface_classifier/results').resolve()
    assert results_dir.is_dir()
    now = datetime.utcnow().strftime(
        '%Y%m%d_%H%M%SZ')     # timestamp string used for traceability
    save_dir = results_dir / now
    save_dir.mkdir(parents=False, exist_ok=False)

    # Log epoch + validation loss to CSV
    #logger = CSVLogger(str(save_dir))

    # Labels and weights
    weights_df = pd.read_csv(
        '/data/road_surface_classifier/dataset_multiclass/class_weights.csv')
    labels = list(weights_df['class_name'])
    top_level_map = list(weights_df['top_level'])
    class_weights = list(weights_df['weight'])

    # Get dataset
    chip_size=224
    preprocess = PreProcess()
    train_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset_multiclass/dataset_train.csv',
        transform=preprocess,
        chip_size=chip_size,
        limit=-1 if not QUICK_TEST else 500,
        n_channels=4 if INCLUDE_NIR else 3)
    val_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset_multiclass/dataset_val.csv',
        transform=preprocess,
        chip_size=chip_size,
        limit=-1 if not QUICK_TEST else 500,
        n_channels=4 if INCLUDE_NIR else 3)

    # Create data loaders.
    batch_size = 64
    train_dl = DataLoader(train_ds,
                          num_workers=16,
                          batch_size=batch_size,
                          shuffle=True)
    val_dl = DataLoader(val_ds, num_workers=16, batch_size=batch_size)

    # Model
    learning_rate=0.00040984645874638675
    model = PLMaskCNN(weights=class_weights,
                      labels=labels,
                      nc=4 if INCLUDE_NIR else 3,
                      top_level_map=top_level_map,
                      learning_rate=learning_rate,
                      seg_k=0.7,
                      ob_k=0.9)


    # Save model to results directory
    torch.save(model, save_dir / 'model.pth')

    # Attempt deserialization (b/c) I've had problems with it before
    try:
        torch.load(save_dir / 'model.pth')
    except:
        import traceback
        traceback.print_exc()
        raise AssertionError('Torch model failed to deserialize!')

    # Train model in stages
    best_model_path: Optional[str] = None
    mlflow_logger: Optional[MLFlowLogger] = None
    stage = 0
    model.set_stage(stage, learning_rate)
    

    # Logger
    mlflow_logger = MLFlowLogger(experiment_name='road_surface_classifier',
                                    run_name='run_%s_%d' % (now, stage),
                                    tracking_uri='http://truenas.local:9809')

    # Upload base model
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id,
                                            str(save_dir / 'model.pth'))

    # Save checkpoints (model states for later)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        monitor='val_loss',
        save_top_k=3,
        filename='model-%d-{epoch:02d}-{val_loss:.5f}' % stage)

    # Setup early stopping based on validation loss
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            mode='min',
                                            patience=10)

    # Stochastic Weight Averaging
    swa_callback = StochasticWeightAveraging(swa_lrs=0.6863423660749621)

    # Trainer
    trainer = pl.Trainer(accelerator='gpu',
                            devices=1,
                            max_epochs=1000 if not QUICK_TEST else 1,
                            callbacks=[
                                checkpoint_callback, early_stopping_callback,
                                swa_callback
                            ],
                            logger=mlflow_logger)

    # Do the thing!
    trainer.fit(model, train_dataloaders=train_dl,
                val_dataloaders=val_dl)     # type: ignore

    # Get best model path
    best_model_path = checkpoint_callback.best_model_path

    # Upload best model
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id,
                                            best_model_path)

    # Load model at best checkpoint for next stage
    del model     # just to be safe
    model = PLMaskCNN.load_from_checkpoint(best_model_path)

    assert best_model_path is not None
    assert mlflow_logger is not None

    # Generate artifacts
    print('Generating artifacts...')
    model.eval()

    # Generate artifacts from model
    from rsc.artifacts import ArtifactGenerator
    from rsc.artifacts.confusion_matrix_handler import ConfusionMatrixHandler
    from rsc.artifacts.accuracy_obsc_handler import AccuracyObscHandler
    from rsc.artifacts.obsc_compare_handler import ObscCompareHandler
    from rsc.artifacts.samples_handler import SamplesHandler
    artifacts_dir = save_dir / 'artifacts'
    artifacts_dir.mkdir(parents=False, exist_ok=True)
    generator = ArtifactGenerator(artifacts_dir, model, val_dl)
    generator.add_handler(ConfusionMatrixHandler(simple=False))
    generator.add_handler(ConfusionMatrixHandler(simple=True))
    generator.add_handler(AccuracyObscHandler())
    generator.add_handler(ObscCompareHandler())
    generator.add_handler(SamplesHandler())
    generator.run(raise_on_error=False)
    mlflow_logger.experiment.log_artifacts(mlflow_logger.run_id,
                                           str(artifacts_dir))

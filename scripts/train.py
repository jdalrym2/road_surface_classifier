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

from rsc.model.plmcnn import PLMaskCNN
from rsc.model.preprocess import PreProcess
from rsc.model.road_surface_dataset import RoadSurfaceDataset
from rsc.artifacts.confusion_matrix_handler import ConfusionMatrixHandler

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
        '/data/road_surface_classifier/dataset/class_weights.csv')

    # NOTE: We add obscuration class with a weight of 1
    labels = list(weights_df['class_name']) + ['obscured']
    class_weights = list(weights_df['weight']) + [1]

    # Get dataset
    preprocess = PreProcess()
    train_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset/dataset_train.csv',
        transform=preprocess,
        limit=-1 if not QUICK_TEST else 500)
    val_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset/dataset_val.csv',
        transform=preprocess,
        limit=-1 if not QUICK_TEST else 500)

    # Create data loaders.
    batch_size = 64
    train_dl = DataLoader(train_ds,
                          num_workers=16,
                          batch_size=batch_size,
                          shuffle=True)
    val_dl = DataLoader(val_ds, num_workers=16, batch_size=batch_size)

    # Model
    model = PLMaskCNN(weights=class_weights,
                      labels=labels,
                      learning_rate=(1e-5),
                      staging_order=(0, ))

    import pickle
    with open(
            '/data/road_surface_classifier/results/20230128_175345Z/stage_1_state_dict.pkl',
            'rb') as f:
        encoder_dict, decoder_dict = pickle.load(f)
    model.model.encoder.load_state_dict(encoder_dict)
    model.model.decoder.load_state_dict(decoder_dict)

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
    for stage, lr in zip(model.staging_order, model.learning_rate):
        print('Training stage %d...' % stage)

        # Set stage, which freezes / unfreezes layers
        model.set_stage(stage, lr)

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
        swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

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
    generator.add_handler(ConfusionMatrixHandler())
    generator.add_handler(AccuracyObscHandler())
    generator.add_handler(ObscCompareHandler())
    generator.add_handler(SamplesHandler())
    generator.run(raise_on_error=False)
    mlflow_logger.experiment.log_artifacts(mlflow_logger.run_id,
                                           str(artifacts_dir))

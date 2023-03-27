#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from typing import Optional

sys.path.append(
    os.getcwd())     # TODO: this is silly, would be fixed with pip install

# Set AWS profile (for use in MLFlow)
os.environ["AWS_PROFILE"] = 'truenas'
os.environ["MLFLOW_S3_ENDPOINT_URL"] = 'http://truenas:9807'

import pathlib
from datetime import datetime

import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import MLFlowLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from rsc.model.plmcnn import PLMaskCNN
from rsc.model.preprocess import PreProcess
from rsc.model.road_surface_dataset import RoadSurfaceDataset

QUICK_TEST = True


def objective(trial: optuna.trial.Trial) -> float:

    # Hyperparameters
    chip_size = trial.suggest_categorical(
        "chip_size", [28, 32, 48, 56, 64, 112, 128, 224, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)
    swa_lrs = trial.suggest_float("swa_lrs", 1e-4, 1e0, log=True)
    loss_lambda = trial.suggest_float("loss_lambda", 0.1, 1, step=0.1)

    # Create directory for results
    results_dir = pathlib.Path(
        '/data/road_surface_classifier/results').resolve()
    assert results_dir.is_dir()
    now = datetime.utcnow().strftime(
        '%Y%m%d_%H%M%SZ')     # timestamp string used for traceability
    save_dir = results_dir / now
    save_dir.mkdir(parents=False, exist_ok=False)

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
        chip_size=chip_size,
        limit=-1 if not QUICK_TEST else 1500)
    val_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset/dataset_val.csv',
        transform=preprocess,
        chip_size=chip_size,
        limit=-1 if not QUICK_TEST else 500)

    # Create data loaders.
    batch_size = 64
    train_dl = DataLoader(train_ds,
                          num_workers=0,
                          batch_size=batch_size,
                          shuffle=True)
    val_dl = DataLoader(val_ds, num_workers=0, batch_size=batch_size)

    # Model
    model = PLMaskCNN(trial=trial,
                      weights=class_weights,
                      labels=labels,
                      learning_rate=(learning_rate, ),
                      loss_lambda=loss_lambda,
                      staging_order=(0, ))

    # Save model to results directory
    torch.save(model, save_dir / 'model.pth')

    # Attempt deserialization (b/c) I've had problems with it before
    try:
        torch.load(save_dir / 'model.pth')
    except:
        traceback.print_exc()
        raise AssertionError('Torch model failed to deserialize!')

    # Train model in stages
    best_model_path: Optional[str] = None
    best_val_loss = float('inf')
    mlflow_logger: Optional[MLFlowLogger] = None
    for stage, lr in zip(model.staging_order, model.learning_rate):
        print('Training stage %d...' % stage)

        # Set stage, which freezes / unfreezes layers
        model.set_stage(stage, lr)

        # Logger
        mlflow_logger = MLFlowLogger(
            experiment_name='road_surface_classifier_optuna_qt2',
            run_name='run_%s_%d_trial_%d' % (now, stage, trial.number),
            tracking_uri='http://truenas:9809')

        # Upload base model
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id,
                                              str(save_dir / 'model.pth'))

        # Save checkpoints (model states for later)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(save_dir),
            monitor='val_loss',
            save_top_k=1,
            filename='model-%d-{epoch:02d}-{val_loss:.5f}' % stage)

        # Setup early stopping based on validation loss
        early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                mode='min',
                                                patience=25)

        # Stochastic Weight Averaging
        swa_callback = StochasticWeightAveraging(swa_lrs=swa_lrs)

        # Trainer
        trainer = pl.Trainer(accelerator='gpu',
                             devices=1,
                             max_epochs=300,
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
        best_val_loss = model.min_val_loss_cl

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
    # from rsc.artifacts.samples_handler import SamplesHandler
    from rsc.artifacts.auc_handler import AUCHandler
    artifacts_dir = save_dir / 'artifacts'
    artifacts_dir.mkdir(parents=False, exist_ok=True)
    generator = ArtifactGenerator(artifacts_dir, model, val_dl)
    generator.add_handler(ConfusionMatrixHandler())
    generator.add_handler(AccuracyObscHandler())
    generator.add_handler(ObscCompareHandler())
    # generator.add_handler(SamplesHandler())
    auc_handler = AUCHandler()
    generator.add_handler(auc_handler)
    generator.run(raise_on_error=False)
    mlflow_logger.experiment.log_artifacts(mlflow_logger.run_id,
                                           str(artifacts_dir))

    return best_val_loss


if __name__ == '__main__':

    now = datetime.utcnow().strftime(
        '%Y%m%d_%H%M%SZ')     # timestamp string used for traceability
    study_name = 'study_%s' % now

    for _ in range(101):
        try:
            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                   n_warmup_steps=5,
                                                   interval_steps=1,
                                                   n_min_trials=5),
                storage='sqlite:///./optuna_qt.sqlite3',
                study_name=study_name,
                load_if_exists=True)
            study.optimize(objective, n_trials=100)
        except Exception:
            traceback.print_exc()
            print('Restarting study...')
        else:
            break

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
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

from rsc.train.plmcnn import PLMaskCNN
from rsc.train.preprocess import PreProcess
from rsc.train.dataset import RoadSurfaceDataset

# Recommended for CUDA
torch.set_float32_matmul_precision('medium')

# Whether or not to include the NIR channel
# since the input dataset includes it
INCLUDE_NIR = True

# Quick test, train an epic on fewer-than-normal
# set of images to check everything works
QUICK_TEST = False

if __name__ == '__main__':

    # Optuna config
    num_trials = 50
    now = datetime.utcnow().strftime(
        '%Y%m%d_%H%M%SZ')     # timestamp string used for traceability
    study_name = 'study_%s' % now

    # Labels and weights
    weights_df = pd.read_csv(
        '/data/road_surface_classifier/dataset_multiclass/class_weights.csv')
    labels = list(weights_df['class_name'])
    top_level_map = list(weights_df['top_level'])
    class_weights = list(weights_df['weight'])

    # Get dataset
    preprocess = PreProcess()
    train_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset_multiclass/dataset_train.csv',
        transform=preprocess,
        chip_size=224,
        limit=-1 if not QUICK_TEST else 1500,
        n_channels=4 if INCLUDE_NIR else 3)
    val_ds = RoadSurfaceDataset(
        '/data/road_surface_classifier/dataset_multiclass/dataset_val.csv',
        transform=preprocess,
        chip_size=224,
        limit=-1 if not QUICK_TEST else 500,
        n_channels=4 if INCLUDE_NIR else 3)

    def objective(trial: optuna.trial.Trial):
        global train_ds, val_ds

        # Create data loaders.
        batch_size = 64
        train_dl = DataLoader(train_ds,
                          num_workers=16,
                          batch_size=batch_size,
                          shuffle=True)
        val_dl = DataLoader(val_ds, num_workers=16, batch_size=batch_size)

        # Hyperparameters
        chip_size = 224
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        swa_lrs = trial.suggest_float("swa_lrs", 1e-4, 1e0, log=True)
        seg_k = trial.suggest_float("seg_k", 0.1, 1, step=0.1)
        ob_k = trial.suggest_float("ob_k", 0.1, 1, step=0.1)

        # Set dataset parameters
        train_ds.set_chip_size(chip_size)
        val_ds.set_chip_size(chip_size)

        # Set model parameters
        model = PLMaskCNN(weights=class_weights,
                      labels=labels,
                      nc=4 if INCLUDE_NIR else 3,
                      top_level_map=top_level_map,
                      learning_rate=learning_rate,
                      seg_k=seg_k,
                      ob_k=ob_k)

        # Create directory for results
        results_dir = pathlib.Path(
            '/data/road_surface_classifier/results').resolve()
        assert results_dir.is_dir()
        now = datetime.utcnow().strftime(
            '%Y%m%d_%H%M%SZ')     # timestamp string used for traceability
        save_dir = results_dir / now
        save_dir.mkdir(parents=False, exist_ok=False)

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
        mlflow_logger: Optional[MLFlowLogger] = None
        stage = 0

        # Logger
        mlflow_logger = MLFlowLogger(
            experiment_name='road_surface_classifier',
            run_name='run_%s_%d_trial_%d' % (now, stage, trial.number),
            tracking_uri='http://truenas:9809')
        mlflow_logger.log_hyperparams(dict(chip_size=chip_size, swa_lrs=swa_lrs))

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
                                                patience=10)

        # Stochastic Weight Averaging
        swa_callback = StochasticWeightAveraging(swa_lrs=swa_lrs)

        # Trainer
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=300,
            callbacks=[checkpoint_callback, early_stopping_callback, swa_callback],
            logger=mlflow_logger)

        # Do the thing!
        trainer.fit(model, train_dataloaders=train_dl,
                    val_dataloaders=val_dl)     # type: ignore

        # Get best model path
        best_model_path = checkpoint_callback.best_model_path

        # Upload best model
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id,
                                            best_model_path)
        
        # Objective return values
        ret = (model.min_val_loss_cl, model.min_val_loss_ob)

        # Load model at best checkpoint to generate artifacts
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
        artifacts_dir = save_dir / 'artifacts'
        artifacts_dir.mkdir(parents=False, exist_ok=True)
        generator = ArtifactGenerator(artifacts_dir, model, val_dl)
        generator.add_handler(ConfusionMatrixHandler(simple=False))
        generator.add_handler(ConfusionMatrixHandler(simple=True))
        generator.add_handler(AccuracyObscHandler())
        generator.add_handler(ObscCompareHandler())
        generator.run(raise_on_error=False)
        mlflow_logger.experiment.log_artifacts(mlflow_logger.run_id,
                                            str(artifacts_dir))

        return ret

    # Do the work!
    for _ in range(3):
        try:
            study = optuna.create_study(
                directions=['minimize', 'minimize'],
                storage='sqlite:///./optuna_rsc.sqlite3',
                study_name='study_20240108_000000Z',
                load_if_exists=True)
            study.optimize(objective, n_trials=num_trials)
        except Exception:
            traceback.print_exc()
            print('Restarting study...')
        else:
            break

    print('Done!')
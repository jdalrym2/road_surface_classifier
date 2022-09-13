#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

#from plresnet50 import PLResnet50
from plmcnn import PLMaskCNN

from data_augmentation import PreProcess
from custom_image_dataset import RoadSurfaceDataset
from handle_metrics import generate_plots

if __name__ == '__main__':

    # Create directory for results
    results_dir = pathlib.Path('../results').resolve()
    assert results_dir.is_dir()
    save_dir = results_dir / datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')
    save_dir.mkdir(parents=False, exist_ok=False)

    # Save checkpoints (model states for later)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        monitor='val_loss',
        save_top_k=10,
        filename='model-{epoch:02d}-{val_loss:.5f}')

    # Setup early stopping based on validation loss
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            mode='min',
                                            patience=25)

    # Log epoch + validation loss to CSV
    logger = CSVLogger(str(save_dir))

    # Get dataset
    preprocess = PreProcess()
    train_ds = RoadSurfaceDataset(
        '/data/road_surface_detection/dataset_simple/dataset_train.csv',
        transform=preprocess)
    val_ds = RoadSurfaceDataset(
        '/data/road_surface_detection/dataset_simple/dataset_val.csv',
        transform=preprocess)

    # Create data loaders.
    batch_size = 32
    train_dl = DataLoader(train_ds,
                          num_workers=16,
                          batch_size=batch_size,
                          shuffle=True)
    val_dl = DataLoader(val_ds, num_workers=16, batch_size=batch_size)

    # Model
    model = PLMaskCNN()
    model.load_from_checkpoint(
        '/home/jon/git/road_surface_detection/results/20220911_012659Z/model-epoch=04-val_loss=0.08898.ckpt'
    )
    torch.save(model, save_dir / 'model.pth')

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger)

    # Do the thing!
    try:
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    except KeyboardInterrupt:
        print('Keyboard interrupt caught!')
        pass

    # Handle metrics
    generate_plots(save_dir, val_dl)

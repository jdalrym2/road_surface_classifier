#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

if __name__ == '__main__':

    model = torch.load(
        '/home/jon/git/road_surface_classifier/results/20220918_193931Z/model.pth'
    ).load_from_checkpoint(
        '/home/jon/git/road_surface_classifier/results/20220918_193931Z/model-epoch=27-val_loss=0.48987.ckpt'
    ).model
    model.eval()

    dummy_input = torch.randn(1, 4, 256, 256, device="cpu")

    # Export the model
    torch.onnx.export(
        model,     # model being run
        dummy_input,     # model input (or a tuple for multiple inputs)
        "/data/road_surface_classifier/best_model.onnx",     # where to save the model (can be a file or file-like object)
        export_params=
        True,     # store the trained parameter weights inside the model file
        opset_version=11,     # the ONNX version to export the model to
        do_constant_folding=
        True,     # whether to execute constant folding for optimization
        input_names=['input'],     # the model's input names
        output_names=['output', 'output2'],     # the model's output names
        dynamic_axes={
            'input': {
                0: 'batch_size'
            },     # variable length axes
            'output': {
                0: 'batch_size'
            }
        })

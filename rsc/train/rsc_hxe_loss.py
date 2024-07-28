#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Hierarchical Cross Entropy Loss """

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSCHXELoss(nn.Module):
    """ Hierarchical Cross Entropy Loss (tailored to RSC model to handle obscuration estimates)"""

    def __init__(self, lv_b_a: torch.Tensor, lv_b_w: torch.Tensor):
        """
        Init the loss function, describing a hierarchy

            (A1)          (A2)       <---- "Level A" | i.e. top level classes; e.g. 'paved' vs 'unpaved')  |
          //    \\      //    \\
        (B1)     (B2) (B3)     (B4)  <---- "Level B" | i.e. prediction classes                             |
                                                     | e.g. 'asphalt' vs 'concrete' vs 'dirt' vs 'gravel') |

        Args:
            lv_b_a (torch.Tensor): Mapping for each label to index of top-level (level A) hierarchy index
                e.g. from above example: (0, 0, 1, 1)
            lv_b_w (torch.Tensor): Weights for prediction (level B) classes, independent of hierarchy
        """        

        super().__init__()

        # Sanity check
        assert len(lv_b_a) == len(lv_b_w)
        device = lv_b_a.device

        # Persist Level B -> Level A mapping
        self.lv_b_a = lv_b_a

        # List of indices between Lv A elements and Lv B elements
        # NOTE: assumes we have a lv B node attached to every lv A node
        self.lv_a_idx = [torch.where(lv_b_a == e)[0] for e in range(int(lv_b_a.max() + 1))]

        # Proportion of Level A elements among level B
        lv_a_p = torch.tensor([(1 / len(lv_b_w) / lv_b_w[idx]).sum() for idx in self.lv_a_idx])

        # Level A weights
        self.lv_a_w = ((1 / len(lv_a_p)) / lv_a_p).to(device)

        # Level B weights for each node in level A
        self.lv_b_w_a = torch.concat([(1 / lv_b_w[idx]).sum() * lv_b_w[idx] / len(idx) for idx in self.lv_a_idx])

    def forward(self, logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss given the model's classification results.

        Args:
            logits (torch.Tensor): Classification model output
            truth (torch.Tensor): Truth classification (one-hot encoded) Shape: (N, num_lv_b)

        Returns:
            torch.Tensor: Computed loss for the model
        """

        # Compute the logarithm of the predicted probabilities
        log_y_pred = F.log_softmax(logits, dim=1)
        
        # Initialize the loss
        loss_lv_b = torch.Tensor((0,)).to(logits.device)
        loss_lv_a = torch.Tensor((0,)).to(logits.device)
        
        # Iterate over the categories
        for i in range(truth.shape[1]):
            # Compute the weight for the category
            w = self.lv_b_w_a[i]

            # Compute the weight for the upper layer
            w2 = self.lv_a_w[self.lv_b_a[i]]
            
            # Compute the cross entropy loss for the category
            cross_entropy = -torch.sum(truth[:, i] * log_y_pred[:, i])
            
            # Add the weighted cross entropy loss to the total loss
            loss_lv_b += w * w2 * cross_entropy

        loss_lv_b /= truth.shape[1]

        # Compute level A logits
        logits_lv_a = torch.stack([torch.sum(logits[:, e], 1) for e in self.lv_a_idx], -1)
        log_y_pred_lv_a = F.log_softmax(logits_lv_a, dim=1)
        truth_lv_a = torch.stack([torch.sum(truth[:, e], 1) for e in self.lv_a_idx], -1)

        # Iterate over the categories
        for i in range(truth_lv_a.shape[1]):
            # Compute the weight for the category
            w = self.lv_a_w[i]
            
            # Compute the cross entropy loss for the category
            cross_entropy = -torch.sum(truth_lv_a[:, i] * log_y_pred_lv_a[:, i])
            
            # Add the weighted cross entropy loss to the total loss
            loss_lv_a += w * cross_entropy

        loss_lv_a /= truth_lv_a.shape[1]

        return (loss_lv_b + loss_lv_a) / truth.shape[0]
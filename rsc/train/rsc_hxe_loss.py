#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Hierarchical Cross Entropy Loss """

import torch
import torch.nn as nn


class RSCHXELoss(nn.Module):
    """ Hierarchical Cross Entropy Loss (tailed to RSC model to handle obscuration estimates)"""

    def __init__(self, lv_b_a: list[float] | torch.Tensor, lv_b_w: list[float] | torch.Tensor):
        """
        Init the loss function, describing a hierarchy

            (A1)          (A2)       <---- "Level A" | i.e. top level classes; e.g. 'paved' vs 'unpaved')  |
          //    \\      //    \\
        (B1)     (B2) (B3)     (B4)  <---- "Level B" | i.e. prediction classes                             |
                                                     | e.g. 'asphalt' vs 'concrete' vs 'dirt' vs 'gravel') |

        Args:
            lv_b_a (list[float] | torch.Tensor): Mapping for each label to index of top-level (level A) hierarchy index
                e.g. from above example: (0, 0, 1, 1)
            lv_b_w (list[float] | torch.Tensor): Weights for prediction (level B) classes, independent of hierarchy
        """        

        super().__init__()

        # Sanity check
        assert len(lv_b_a) == len(lv_b_w)

        # Level B -> Level A mapping
        lv_b_a = torch.Tensor(lv_b_a)

        # Level B weights
        lv_b_w = torch.Tensor(lv_b_w)

        # List of indices between Lv A elements and Lv B elements
        # NOTE: assumes we have a lv B node attached to every lv A node
        self.lv_a_idx = [torch.where(lv_b_a == e)[0] for e in range(lv_b_a.max() + 1)]

        # Proportion of Level A elements among level B
        lv_a_p = torch.tensor([(1 / len(lv_b_w) / lv_b_w[idx]).sum() for idx in self.lv_a_idx])

        # Level A weights
        # Add (1) for obscuration loss
        self.lv_a_w = torch.concat(((1 / len(lv_a_p)) / lv_a_p, torch.Tensor((1,))))

        # Level B weights for each node in level A
        # Add (1) for obscuration loss
        lv_b_w_a = [(1 / lv_b_w[idx]).sum() * lv_b_w[idx] / len(idx) for idx in self.lv_a_idx] + [torch.Tensor((1,))]

        # Add custom indices for obscuration estimates to Lv A (RSC-specific)
        self.lv_a_idx += [[-2, -1]]

        # Create loss classes for each level B cluster and assign weights
        self.lv_b_loss = [torch.nn.BCEWithLogitsLoss(w) for w in lv_b_w_a]


    def forward(self, logits: torch.Tensor, truth_c: torch.Tensor, truth_obsc: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss given the model's classification results.

        Args:
            logits (torch.Tensor): Classification model output
            truth_c (torch.Tensor): Truth classification (one-hot enocded) Shape: (N, num_lv_b)
            truth_obsc (torch.Tensor): Truth obscuration percentage (shape (N, 1))

        Returns:
            torch.Tensor: Computed loss for the model
        """

        # Combine truth
        truth = torch.concat((truth_c, truth_obsc, 1 - truth_obsc), 1)

        # Compute BCE loss for each Lv B cluster
        b_loss = torch.Tensor([loss(logits[:, idx], truth[:, idx]) for loss, idx in zip(self.lv_b_loss, self.lv_a_idx)])

        # Final loss is weighted average based on Lv A weights
        loss = (self.lv_a_w * b_loss).sum()

        return loss

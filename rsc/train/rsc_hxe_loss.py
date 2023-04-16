#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Hierarchical Cross Entropy Loss """

import torch
import torch.nn as nn


class RSCHXELoss(nn.Module):
    """ Hierarchical Cross Entropy Loss (tailed to RSC model to handle obscuration estimates)"""

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

        # List of indices between Lv A elements and Lv B elements
        # NOTE: assumes we have a lv B node attached to every lv A node
        self.lv_a_idx = [torch.where(lv_b_a == e)[0] for e in range(int(lv_b_a.max() + 1))]

        # Proportion of Level A elements among level B
        lv_a_p = torch.tensor([(1 / len(lv_b_w) / lv_b_w[idx]).sum() for idx in self.lv_a_idx])

        # Level A weights
        self.lv_a_w = ((1 / len(lv_a_p)) / lv_a_p).to(device)

        # Level B weights for each node in level A
        lv_b_w_a = [(1 / lv_b_w[idx]).sum() * lv_b_w[idx] / len(idx) for idx in self.lv_a_idx]

        # Create loss classes for each level B cluster and assign weights
        # self.lv_b_loss = [torch.nn.BCEWithLogitsLoss(w.to(device)) for w in lv_b_w_a]
        self.lv_b_loss = [torch.nn.CrossEntropyLoss(w.to(device), reduction='mean') for w in lv_b_w_a]

        # Create loss for each level A cluster and assign weights
        self.lv_a_loss = torch.nn.CrossEntropyLoss(self.lv_a_w, reduction='mean')

        # Loss accounting for obscuration (no weights)
        self.o_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')


    def forward(self, logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss given the model's classification results.

        Args:
            logits (torch.Tensor): Classification model output
            truth (torch.Tensor): Truth classification (one-hot encoded + obsc + 1 - obsc) Shape: (N, num_lv_b + 2)

        Returns:
            torch.Tensor: Computed loss for the model
        """

        # Compute BCE loss for each Lv B cluster
        b_loss = torch.Tensor([loss(logits[:, idx], truth[:, idx]) for loss, idx in zip(self.lv_b_loss, self.lv_a_idx)]).to(logits.device).sum()

        # Aggregate truth and logits of level a
        logits_a = torch.stack([torch.sum(logits[:, idx], 1) for idx in self.lv_a_idx], -1).to(logits.device)
        truth_a = torch.stack([torch.sum(truth[:, idx], 1) for idx in self.lv_a_idx], -1).to(logits.device)

        # Compute BCE loss for each level A cluster
        a_loss = self.lv_a_loss(logits_a, truth_a)

        # Compute BCE loss for obscuration
        o_loss = self.o_loss(logits[:, (-2, -1)], truth[:, (-2, -1)])

        # Final loss is sum of these
        return 0.1 * b_loss + 0.9 * a_loss + 2 * o_loss

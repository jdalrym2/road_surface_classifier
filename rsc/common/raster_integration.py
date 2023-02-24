#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np


def transverse_grid(x1: float, y1: float, x2: float,
                    y2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a vector line, determine the pixels / grid cells
    the line crosses as well as the length of the line in each cell.

    Inspired by "A Fast Voxel Transversal Algorithm for Ray Tracing"
    Link: http://www.cse.yorku.ca/~amana/research/grid.pdf

    Args:
        x1 (float): Line start x- coordinate
        y1 (float): Line start y- coordinate
        x2 (float): Line end x- coordinate
        y2 (float): Line end y- coordinate

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays:
            - (N, 2) array of grid locations the line intersects
            - (N,) array of lengths of the line in each pixel
    """

    # Holds pixel coordinates the line crosses
    # and lengths of the line in each pixel
    coords, lengths = [], []

    # Parameterize line
    # [x, y]^T = [x1, y1]^T + t*[dx, dy]^T
    # t \in [0, 1]
    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    step_x = 1 if x2 > x1 else -1
    step_y = 1 if y2 > y1 else -1
    ell = (dx**2 + dy**2)**0.5     # length of the line

    # Init state variables
    x, y = float(x1), float(y1)

    # Where (in t- coordinates) is the first vertical pixel boundary crossed?
    # Where (in t- coordinates) is the first horizontal pixel boundary crossed?
    # If any of these are after the end of the line, we truncate them
    t_max_x = abs(np.floor(x + (step_x > 0)) - x) / dx if dx > 0 else np.inf
    t_max_x = min(1, t_max_x) if dx > 0 else t_max_x
    t_max_y = abs(np.floor(y + (step_y > 0)) - y) / dy if dy > 0 else np.inf
    t_max_y = min(1, t_max_y) if dy > 0 else t_max_y

    # How far do we have to go (units of t) to equate to the width
    # of one pixel? The height?
    t_delta_x = 1 / dx if dx > 0 else None
    t_delta_y = 1 / dy if dy > 0 else None
    assert not (t_delta_x is None and t_delta_y is None)

    # Coords and length of line for first pixel
    coords.append((np.floor(x), np.floor(y)))
    lengths.append(ell * np.min((t_max_x, t_max_y)))

    # Iterate over the length of the line
    while t_max_x < 1 or t_max_y < 1:
        incr_x, incr_y = 0, 0

        # Determine the amount to increment
        # Usually we increment *either* x or y
        # But for a 45 degree line that intersects
        # 4 corners regions we actually increment both
        if t_delta_x and t_max_x <= t_max_y:
            incr_x = min(t_delta_x, 1 - t_max_x)
        if t_delta_y and t_max_y <= t_max_x:
            incr_y = min(t_delta_y, 1 - t_max_y)

        # Actually increment the state variables
        if incr_x:
            t_max_x += incr_x
            x += step_x
            lengths.append(incr_x * ell)
        if incr_y:
            t_max_y += incr_y
            y += step_y
            lengths.append(incr_y * ell)

        # If I incremented both, then we appended the lengths
        # twice, so pop the first
        if incr_x and incr_y:
            lengths.pop()

        # Append the pixel coordinate we're in
        coords.append((np.floor(x), np.floor(y)))

    # Return result as numpy array
    return np.array(coords, dtype=int), np.array(lengths)

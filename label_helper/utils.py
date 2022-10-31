#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import PIL.Image


def image_to_pixmap(image_path: pathlib.Path) -> QPixmap:
    """
    Read in an image and get a QPixmap from that image

    Args:
        image_path (pathlib.Path): Image path

    Returns:
        QPixmap: Qt pixel map
    """
    # NOTE: this seems roundabout since Qt has an image reader
    # but the benefit of this is we can modify the array (e.g.
    # make it 8-bit) before passing it to the pixmap!
    assert image_path.is_file()
    im = np.array(PIL.Image.open(image_path))
    if im.ndim == 2:
        h, w = im.shape
        c = 1
        format = QImage.Format_Grayscale8
    else:
        h, w, c = im.shape
        format = QImage.Format_RGB888
    q_img = QImage(im, w, h, c * w, format)
    return QPixmap.fromImage(q_img)

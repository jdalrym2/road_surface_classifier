#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Road Surface Classifier Labeling Helper Tool """

import argparse
import pathlib
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from .controller import Controller
from .ui.RSC_LabelHelper import Ui_MainWindow


def parse_args():
    parser = argparse.ArgumentParser(
        'python3 -m label_helper',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--input-path',
                        type=str,
                        required=True,
                        help='Input path to SQLite file')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    input_path = pathlib.Path(args.input_path)
    assert input_path.is_file()

    # Set up application and main window
    app = QApplication([])
    win = QMainWindow()
    win.ui = Ui_MainWindow()
    win.ui.setupUi(win)

    # Init controller
    Controller(win.ui, input_path)

    # Launch the GUI!
    win.show()
    sys.exit(app.exec_())
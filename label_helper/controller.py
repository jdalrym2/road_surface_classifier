#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import enum

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtGui import QPainter, QPixmap

from .basic_attr_model import BasicAttrModel
from .label_attr_model import LabelAttrModel

from .sqlite3_interface import Sqlite3Interface
from .ui.RSC_LabelHelper import Ui_MainWindow
from .utils import image_to_pixmap


class KeyEnum(enum.Enum):
    OBSCURATION = 'Obscuration'
    NO_DATA = 'No Data'
    BAD_DETECT = 'Bad Detect'


class Controller:

    def __init__(self, ui: Ui_MainWindow, sqlite3_path: pathlib.Path):

        # Sanity check input
        assert sqlite3_path.is_file()

        # Sqlite Interface
        self.interface = Sqlite3Interface(sqlite3_path)

        # Persist main window ui
        self.ui = ui

        # State variables
        self.cur_image_idx = None
        self.num_features = self.interface.get_num_features()
        self.chip_path = None
        self.mask_path = None

        # Models
        self.basicAttrModel = BasicAttrModel()
        self.labelAttrModel = LabelAttrModel()
        self.ui.basicAttrTableView.setModel(self.basicAttrModel)
        self.ui.labelAttrTableView.setModel(self.labelAttrModel)

        # UI Configuration
        for it in (self.ui.basicAttrTableView, self.ui.labelAttrTableView):
            it.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.imgNumSlider.setRange(0, self.num_features)
        self.ui.imgNumSpinBox.setRange(0, self.num_features)
        self.ui.imgNumSlider.valueChanged.connect(self._slider_value_changed)
        self.ui.imgNumSlider.sliderReleased.connect(
            lambda: self.load_image_by_index(self.ui.imgNumSlider.value()))
        self.ui.opacityHorizontalSlider.valueChanged.connect(
            lambda v: self.update_img_label())
        self.ui.imgNumSpinBox.valueChanged.connect(
            lambda v: self.load_image_by_index(v))
        self.ui.inputLineEdit.returnPressed.connect(
            self._input_line_edit_text_submitted)

        self.load_image_by_index(0)

    def skip(self, forward=True):
        idx = self.cur_image_idx if self.cur_image_idx is not None else 0
        if forward:
            while idx + 1 < self.num_features:
                feat_dict = self.interface.get_feature(idx + 1)
                if feat_dict.get('obscuration', -1) < 0 and not feat_dict.get(
                        'bad_detect') and not feat_dict.get('no_data'):
                    break
                idx += 1
            idx += 1
        else:
            while idx > 1:
                feat_dict = self.interface.get_feature(idx - 1)
                if feat_dict.get('obscuration', -1) < 0 and not feat_dict.get(
                        'bad_detect') and not feat_dict.get('no_data'):
                    break
                idx -= 1
            idx -= 1

        print('Skiping to index:', idx)

        self.load_image_by_index(idx)

    def load_next_image(self):
        if self.cur_image_idx is not None:
            self.load_image_by_index(self.cur_image_idx + 1)

    def load_prev_image(self):
        if self.cur_image_idx is not None:
            self.load_image_by_index(self.cur_image_idx - 1)

    def load_image_by_index(self, idx: int):
        if not 0 <= idx <= self.num_features:
            return

        if self.cur_image_idx is not None:
            self.save_state()

        self.cur_image_idx = idx

        feature_dict = self.interface.get_feature(idx)
        self.basicAttrModel.add_labels({
            'OSM ID': feature_dict['osm_id'],
            'Class': feature_dict['class']
        })
        self.labelAttrModel.add_labels({
            KeyEnum.OBSCURATION.value:
            feature_dict['obscuration'],
            KeyEnum.NO_DATA.value:
            feature_dict['no_data'],
            KeyEnum.BAD_DETECT.value:
            feature_dict['bad_detect']
        })

        # Set file label
        file_label_text = '<b>Chip Path:&nbsp;&nbsp;</b> %s<br/><b>Mask Path:&nbsp;&nbsp;</b>%s' % (
            str(feature_dict['chip_path']), str(feature_dict['mask_path']))
        self.ui.fileLabel.setText(file_label_text)

        self.chip_path = feature_dict['chip_path']
        self.mask_path = feature_dict['mask_path']

        self.update_img_label()

        self.ui.imgNumSlider.blockSignals(True)
        self.ui.imgNumSlider.setValue(self.cur_image_idx)
        self.ui.imgNumSlider.blockSignals(False)

        self.ui.imgNumSpinBox.blockSignals(True)
        self.ui.imgNumSpinBox.setValue(self.cur_image_idx)
        self.ui.imgNumSpinBox.blockSignals(False)

    def save_state(self):
        # Get obscuration of current index
        obsc = int(
            self.labelAttrModel.attr_dict.get(KeyEnum.OBSCURATION.value, -1))
        nd = int(self.labelAttrModel.attr_dict.get(KeyEnum.NO_DATA.value, 0))
        bd = int(self.labelAttrModel.attr_dict.get(KeyEnum.BAD_DETECT.value,
                                                   0))

        if self.cur_image_idx is not None:
            self.interface.set_feature_obscuration(self.cur_image_idx, obsc)
            self.interface.set_feature_no_data(self.cur_image_idx, bool(nd))
            self.interface.set_feature_bad_detect(self.cur_image_idx, bool(bd))

    def update_img_label(self):
        if self.chip_path is None or self.mask_path is None:
            return

        px1 = image_to_pixmap(self.chip_path)
        px1: QPixmap = px1.scaled(self.ui.imgLabel.width(),
                                  self.ui.imgLabel.height(),
                                  Qt.KeepAspectRatio)

        px2 = image_to_pixmap(self.chip_path)
        px2: QPixmap = px1.scaled(self.ui.imgLabelMask.width(),
                                  self.ui.imgLabelMask.height(),
                                  Qt.KeepAspectRatio)

        px3 = image_to_pixmap(self.mask_path)
        px3 = px3.scaled(px2.width(), px2.height())

        painter = QPainter(px2)
        comp_mode = QPainter.CompositionMode_Screen if self.ui.screenRadioButton.isChecked(
        ) else QPainter.CompositionMode_Source
        painter.setCompositionMode(comp_mode)
        painter.setOpacity(self.ui.opacityHorizontalSlider.value() /
                           self.ui.opacityHorizontalSlider.maximum())
        painter.drawPixmap(0, 0, px2.width(), px2.height(), px3)
        del painter

        self.ui.imgLabel.setPixmap(px1)
        self.ui.imgLabelMask.setPixmap(px2)

    def _slider_value_changed(self, v: int):
        self.ui.imgNumSpinBox.blockSignals(True)
        self.ui.imgNumSpinBox.setValue(v)
        self.ui.imgNumSpinBox.blockSignals(False)

    def _input_line_edit_text_submitted(self):
        v = self.ui.inputLineEdit.text()
        self.ui.inputLineEdit.blockSignals(True)
        if v.endswith('cc'):
            self.ui.inputLineEdit.clear()
        elif v.startswith('g'):
            try:
                g_val = int(v[1:])
            except ValueError:
                pass
            else:
                self.load_image_by_index(g_val)
            finally:
                self.ui.inputLineEdit.clear()
        elif v.startswith('o') and v != 'oo':
            try:
                o_val = int(v[1:])
            except ValueError:
                pass
            else:
                if -1 <= o_val <= 10:
                    self.labelAttrModel.attr_dict[
                        KeyEnum.OBSCURATION.value] = o_val
                    self.labelAttrModel.update_layout()
                    self.load_next_image()
            finally:
                self.ui.inputLineEdit.clear()
        elif v.startswith('t'):
            try:
                t_val = int(v[1:])
            except ValueError:
                pass
            else:
                if 0 <= t_val <= 10:
                    self.ui.opacityHorizontalSlider.setValue(t_val * 10)
            finally:
                self.ui.inputLineEdit.clear()
        elif v == 'p':
            self.load_prev_image()
            self.ui.inputLineEdit.clear()
        elif v == 'n':
            self.load_next_image()
            self.ui.inputLineEdit.clear()
        elif v == 'ps':
            self.skip(forward=False)
            self.ui.inputLineEdit.clear()
        elif v == 'ns':
            self.skip(forward=True)
            self.ui.inputLineEdit.clear()
        elif v == 'nn' or v == 'oo':
            self.labelAttrModel.attr_dict[KeyEnum.OBSCURATION.value] = 0
            self.labelAttrModel.update_layout()
            self.load_next_image()
            self.ui.inputLineEdit.clear()
        elif v == 'bd':
            self.labelAttrModel.attr_dict[
                KeyEnum.BAD_DETECT.value] = 1 - self.labelAttrModel.attr_dict[
                    KeyEnum.BAD_DETECT.value]
            self.labelAttrModel.update_layout()
            self.ui.inputLineEdit.clear()
        elif v == 'dd':
            self.labelAttrModel.attr_dict[
                KeyEnum.NO_DATA.value] = 1 - self.labelAttrModel.attr_dict[
                    KeyEnum.NO_DATA.value]
            self.labelAttrModel.update_layout()
            self.ui.inputLineEdit.clear()

        self.ui.inputLineEdit.blockSignals(False)

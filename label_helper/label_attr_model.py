#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtCore import QModelIndex, Qt

from .basic_attr_model import BasicAttrModel


class LabelAttrModel(BasicAttrModel):

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = super().flags(index)
        if index.column() == 1:
            flags |= Qt.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role) -> bool:
        if not role == Qt.EditRole:
            return False

        if not index.column() == 1:
            return False

        try:
            value = int(value)
        except ValueError:
            return False

        if -1 <= value <= 10:
            k, _ = self.attr_list[index.row()]
            self.attr_dict[k] = value
            self.update_layout()
            return True

        return False

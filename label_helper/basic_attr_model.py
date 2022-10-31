#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt


class BasicAttrModel(QAbstractTableModel):

    __slots__ = ['attr_dict']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_dict: Dict[str, Any] = {}

    @property
    def attr_list(self) -> List[Tuple[str, Any]]:
        return list(self.attr_dict.items())

    def rowCount(self, parent: QModelIndex) -> int:
        if parent.isValid():
            return 0
        return len(self.attr_dict)

    def columnCount(self, parent: QModelIndex) -> int:
        return 2

    def headerData(self, section: int, orientation: Qt.Orientation, role):
        if not role == Qt.DisplayRole:
            return

        if not orientation == Qt.Horizontal:
            return

        return ['Label', 'Value'][section]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = super().flags(index)
        flags |= Qt.ItemIsSelectable
        return flags

    def data(self, index: QModelIndex, role):
        if role == Qt.DisplayRole:
            if 0 <= index.column() <= 1:
                return str(self.attr_list[index.row()][index.column()])

    def setData(self, index: QModelIndex, value, role) -> bool:
        if not role == Qt.EditRole:
            return False

        if not index.column() == 1:
            return False

        k, _ = self.attr_list[index.row()]
        self.attr_dict[k] = value

        return True

    def add_label(self, k: str, v: Any, update_layout: bool = True):
        self.attr_dict[k] = v
        if update_layout:
            self.update_layout()

    def add_labels(self, v: Dict[str, Any], update_layout: bool = True):
        self.attr_dict.update(v)
        if update_layout:
            self.update_layout()

    def clear_state(self, update_layout: bool = True):
        self.attr_dict.clear()
        if update_layout:
            self.update_layout()

    def update_layout(self):
        self.layoutChanged.emit()
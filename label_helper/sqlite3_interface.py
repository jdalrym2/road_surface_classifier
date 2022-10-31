#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import sqlite3
from typing import Any, Dict, List, Tuple, Union


class Sqlite3Interface():
    """ Class to help interface with sqlite3 file we are to label """

    __slots__ = ['input_path', 'verbose', 'class_map']

    def __init__(self, input_path: pathlib.Path, *, verbose: bool = False):
        assert input_path.is_file()
        self.input_path = input_path
        self.verbose = verbose
        self.class_map = self.get_class_map()

    def get_class_map(self) -> Dict[int, str]:
        """
        Get a mapping of class index numbers to class label strings

        Returns:
            Dict[int, str]: Mapping of class num -> class label
        """
        sql, sql_args = self._sql_get_class_map()
        result = self._sql_execute(sql, sql_args, fetch=True)
        return {int(row['class_num']): row['class_name'] for row in result}

    def get_num_features(self) -> int:
        """
        Get the number of features in the DB

        Returns:
            int: Number of features
        """
        sql, sql_args = self._sql_get_num_rows()
        result = self._sql_execute(sql, sql_args, fetch=True)
        num_rows, = result[0]
        return num_rows

    def get_feature(self, index: int) -> Dict[str, Any]:
        """
        Get a feature from the DB based on index

        Args:
            index (int): Index of feature

        Returns:
            Dict[str, Any]: Dictionary describing feature attributes
        """
        sql, sql_args = self._sql_get_row_by_index(index)
        result = self._sql_execute(sql, sql_args, fetch=True)
        row, = result
        return {
            'index': int(row['index']),
            'osm_id': int(row['osm_id']),
            'class': self.class_map.get(int(row['class_num']), 'Unknown'),
            'chip_path': pathlib.Path(row['chip_path']),
            'mask_path': pathlib.Path(row['mask_path']),
            'obscuration': int(row['obscuration']),
            'no_data': int(row['no_data']),
            'bad_detect': int(row['bad_detect'])
        }

    def set_feature_obscuration(self, index: int, obsc: int) -> None:
        """
        Set obscuration of a feature based on index

        Args:
            index (int): Feature index
            obsc (int): Obscuration. Either -1 for 'unspecified'. Or 0 through 10.
        """
        assert -1 <= obsc <= 10
        sql, sql_args = self._sql_set_obscurity_attr(index, obsc)
        self._sql_execute(sql, sql_args, fetch=False)

    def set_feature_no_data(self, index: int, nd: bool) -> None:
        """
        Set "no data" attribute of a feature based on index

        Args:
            index (int): Feature index
            nd (bool): No_data value. True for "no data"
        """
        sql, sql_args = self._sql_set_no_data_attr(index, nd)
        self._sql_execute(sql, sql_args, fetch=False)

    def set_feature_bad_detect(self, index: int, bd: bool) -> None:
        """
        Set "bad detect" attribute of a feature based on index

        Args:
            index (int): Feature index
            bd (bool): "Bad Detect" value. True for "bad detect"
        """
        sql, sql_args = self._sql_set_bad_detect_attr(index, bd)
        self._sql_execute(sql, sql_args, fetch=False)

    def _sql_execute(self,
                     sql: str,
                     sql_args: List[Any],
                     *,
                     fetch: bool = False) -> List[sqlite3.Row]:
        """
        Execute a SQL statement

        Args:
            sql (str): SQL string to execute. Parameters locations should be '?'s
            sql_args (List[Any]): SQL parameters to include
            fetch (bool, optional): Whether or not this statement involves a fetch. Defaults to False.

        Returns:
            List[Any]: List of results. Empty if `fetch == False`.
        """
        ret = []
        with sqlite3.connect(self.input_path) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            if self.verbose:
                print(f'{sql}\t{repr(sql_args)}')
            res = cur.execute(sql, sql_args)
            if fetch:
                ret = res.fetchall()
            cur.close()
        return ret

    @staticmethod
    def _sql_get_class_map() -> Tuple[str, List[Any]]:
        return 'SELECT class_name, class_num from classes;', []

    @staticmethod
    def _sql_get_num_rows() -> Tuple[str, List[Any]]:
        return 'SELECT COUNT(*) from features;', []

    @staticmethod
    def _sql_get_row_by_index(idx: int) -> Tuple[str, List[Any]]:
        return 'SELECT * from features WHERE [index] == ?;', [idx]

    @staticmethod
    def _sql_set_obscurity_attr(idx: int, obsc: int) -> Tuple[str, List[Any]]:
        return 'UPDATE features SET obscuration = ? WHERE [index] = ?;', [
            obsc, idx
        ]

    @staticmethod
    def _sql_set_no_data_attr(idx: int, nd: bool) -> Tuple[str, List[Any]]:
        return 'UPDATE features SET no_data = ? WHERE [index] = ?;', [
            int(nd), idx
        ]

    @staticmethod
    def _sql_set_bad_detect_attr(idx: int, bd: bool) -> Tuple[str, List[Any]]:
        return 'UPDATE features SET bad_detect = ? WHERE [index] = ?;', [
            int(bd), idx
        ]


if __name__ == '__main__':

    sqlite3_path = pathlib.Path(
        '/data/road_surface_classifier/features.sqlite3')
    interface = Sqlite3Interface(sqlite3_path, verbose=True)
    print(interface.class_map)
    print(interface.get_num_features())
    print(interface.get_feature(0))
    interface.set_feature_obscuration(4, 9)
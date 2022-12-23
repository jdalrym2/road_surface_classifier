#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Convert Road Surface Dataset Features to Sqlite for Label Helper Tool """

import pathlib
import argparse
import sqlite3

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--input-path',
                        type=str,
                        required=True,
                        help='Input path to dataset CSV file')
    parser.add_argument('-c',
                        '--class-path',
                        type=str,
                        required=True,
                        help='Input path to dataset class CSV file')
    parser.add_argument('-o',
                        '--output-path',
                        type=str,
                        required=True,
                        help='Output path to SQLite file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    input_path = pathlib.Path(args.input_path)
    assert input_path.is_file()
    class_path = pathlib.Path(args.class_path)
    assert class_path.is_file()
    output_path = pathlib.Path(args.output_path)
    assert not output_path.exists()

    # Read in CSV file
    print('Parsing OSM CSV...')
    osm_df = pd.read_csv(input_path)

    # Keep only columns we want
    osm_df = osm_df[['osm_id', 'class_num', 'chip_path', 'mask_path']]

    # Shuffle!
    osm_df = osm_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add column for obscuration amount
    osm_df = osm_df.assign(obscuration=-1, no_data=0, bad_detect=0)

    # Read in class CSV
    print('Parsing class CSV...')
    class_df = pd.read_csv(class_path)

    # Keep only columns we want
    class_df = class_df[['class_name', 'class_num']]

    print('Saving features to file...')
    with sqlite3.connect(output_path) as con:
        osm_df.to_sql('features', con)
        class_df.to_sql('classes', con)

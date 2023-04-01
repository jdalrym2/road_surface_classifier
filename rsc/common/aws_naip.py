#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Some helpful functions to download NAIP data on AWS """

import pathlib
from typing import List

import boto3

BASE_PATH = pathlib.Path('/data/road_surface_classifier')
assert BASE_PATH.is_dir()
AWS_PATH = BASE_PATH / 'naip_on_aws'
AWS_PATH.mkdir(exist_ok=True)


def naip_s3_fetch(bucket_name: str, object_name: str, output: pathlib.Path):
    """
    Fetch an S3 object from a bucket, ensuring that RequestPayer=requester is set

    Args:
        bucket_name (str): AWS S3 bucket name
        object_name (str): Object name in bucket
        output (pathlib.Path): Path to save the output file

    Raises:
        FileExistsError: If the output file already exists
    """
    # Prevent overwriting the output file
    if output.exists():
        raise FileExistsError('File already exists: %s' % str(output))

    # Use the boto3 s3 API to fetch the file object
    s3_client = boto3.client('s3')
    try:
        with open(output, 'wb') as f:
            s3_client.download_fileobj(bucket_name,
                                       object_name,
                                       f,
                                       ExtraArgs={'RequestPayer': 'requester'})
    except:
        # If cancelled or something bad happens,
        # unlink the file we are trying to download and raise
        output.unlink()
        raise


def get_naip_manifest(bucket_name='naip-analytic') -> List[str]:
    """
    Read the bucket manifest for a given NAIP on AWS bucket. If the manifest
    doesn't yet exist, then fetch it from S3.

    Args:
        bucket_name (str, optional): NAIP on AWS bucket name. Defaults to 'naip-analytic'.

    Returns:
        List[str]: List of objects in manifest
    """
    manifest_path = AWS_PATH / 'manifest.txt'

    # Fetch the manifest if it doesn't exist.
    if not manifest_path.exists():
        naip_s3_fetch(bucket_name, 'manifest.txt', manifest_path)

    # Read it in!
    with open(manifest_path, 'r') as f:
        return [e.strip() for e in f.readlines()]


def get_naip_file(object_name: str,
                  bucket_name='naip-analytic') -> pathlib.Path:
    """
    Get a NAIP file by object name. If it already exists, this will not download again.

    Args:
        object_name (str): Object name
        bucket_name (str, optional): NAIP on AWS Bucket name. Defaults to 'naip-analytic'.

    Returns:
        pathlib.Path: Path to the downloaded file.
    """
    # Get output path, return immediately if exists
    output_path = AWS_PATH / object_name
    if output_path.exists():
        return output_path

    # Otherwise, create the parent directory, and fetch the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    naip_s3_fetch(bucket_name, object_name, output_path)

    return output_path

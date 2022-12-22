#!/bin/bash
TAG=truenas.local:9909/road_surface_classifier:latest  # set this to whatever you want
DOCKER_BUILDKIT=1 docker build -t $TAG .

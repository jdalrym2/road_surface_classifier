#!/bin/bash

# https://hub.docker.com/r/wiktorn/overpass-api
# http://localhost:12345/api/interpreter
docker run \
  -e OVERPASS_META=yes \
  -e OVERPASS_MODE=init \
  -e OVERPASS_PLANET_URL=file:///data/gis/us-latest.osm.bz2 \
  -e OVERPASS_RULES_LOAD=10 \
  -e OVERPASS_SPACE=55000000000 \
  -e OVERPASS_MAX_TIMEOUT=86400 \
  -v /data/gis:/data/gis \
  -v /data/gis/overpass_db:/db \
  -p 12345:80 \
  -i --name overpass_usa wiktorn/overpass-api:latest

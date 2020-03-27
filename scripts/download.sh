#!/bin/bash

set -e

DATA_URL=http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
path=${1:-data}

echo "Extracting image data to $(readlink -f "$path")"
mkdir -p $path
wget -nv --show-progress -O - $DATA_URL | tar -xzC "$path"

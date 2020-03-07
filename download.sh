#!/bin/bash

DATA_URL=http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz

target=${1:-data}

echo "Downloading image data from $DATA_URL"
echo "Extracting to $(readlink -f $target)"
wget -nv --show-progress -O - $DATA_URL | tar -xzC $target

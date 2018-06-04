#!/bin/bash

echo 'Downloading data...'
mkdir data
wget https://polybox.ethz.ch/index.php/s/l2wM4RIyI3pD7Tl/download -O data/train.csv
wget https://polybox.ethz.ch/index.php/s/02IVLdBAgVcsJAx/download -O data/val.csv
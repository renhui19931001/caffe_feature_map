#!/usr/bin/env sh
set -e

./build/tools/extract_features mine/lenet_original.caffemodel mine/lenet_extract.prototxt mine/mnist_train_lmdb ip1 mine/feature_train 600 lmdb CPU 

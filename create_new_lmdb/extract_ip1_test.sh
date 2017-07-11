#!/usr/bin/env sh
set -e

./build/tools/extract_features mine/lenet_original.caffemodel mine/lenet_extract.prototxt mine/mnist_test_lmdb ip1 mine/feature_test 100 lmdb CPU 

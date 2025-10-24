#!/bin/bash

curl -O https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz
curl -O https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz
curl -O https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz
curl -O https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
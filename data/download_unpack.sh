#!/bin/bash

wget https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz
wget https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz
wget https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz
wget https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
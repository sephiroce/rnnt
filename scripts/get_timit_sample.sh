#!/bin/bash

BASE=https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1
wget $BASE.txt
wget ${BASE}.phn
wget ${BASE}.wav
wget ${BASE}.wrd
echo "timit samples were downloaded."
mkdir -p samples/data/timit_sample
mv LDC93S1.* samples/data/timit_sample
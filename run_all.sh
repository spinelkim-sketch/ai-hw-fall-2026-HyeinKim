#!/bin/bash
# Runs all 6 experiments (3 models x with/without augmentation)
for m in mlp cnn transformer; do
  python train.py --model $m
  python train.py --model $m --augment
done
python summarize.py

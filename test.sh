#!/bin/sh
MODEL=ftnet
BACKBONE=resnext101_32x8d
GPUS=2
FILTERS=128
EDGES='3'
NBLOCKS=2
NODES=1
ALPHA=20

DATASET2='soda'
RUNDIR2=/home/ilan/Desktop/Niv/ThermalSegmentation/"$DATASET2"/

## Testing
python  /home/ilan/Desktop/Niv/ThermalSegmentation/Codes/src/lightning_scripts/main.py \
--mode 'test' \
--model "$MODEL" \
--edge-extracts "$EDGES" \
--loss-weight "$ALPHA" \
--num-blocks "$NBLOCKS" \
--backbone "$BACKBONE" \
--no-of-filters "$FILTERS" \
--test-monitor 'val_mIOU' \
--test-monitor-path "$RUNDIR2"/ckpt/ \
--pretrained-base False \
--dataset "$DATASET2" \
--dataset-path '/home/ilan/Desktop/Niv/ThermalSegmentation/processed_dataset/SODA' \
--test-batch-size 1 \
--save-images True \
--save-images-as-subplots False \
--debug False \
--wandb-name-ext '' \
--save-dir "$RUNDIR2"/Best_MIOU/

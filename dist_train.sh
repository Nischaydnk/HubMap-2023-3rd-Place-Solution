#!/usr/bin/env bash

GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \xw
sudo python -W ignore -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=3 \
    --master_port=29500 \
    train.py \
    /home/nischay/hubmap/vitadap/detection/hubconf3/exp4_adapbeitv2l_withps50exp2.py \
    --seed 69 \
    --launcher pytorch ${@:3}

#!/bin/bash

torchrun \
	--nnodes 2 \
	--nproc_per_node 8 \
	--rdzv-id 0 \
	--rdzv-backend c10d \
	--rdzv-endpoint ampere2.stanford.edu:29400 \
	train.py \
	--train-path /lfs/local/0/ranjanr/cc-train/train.bin \
	--dev-path /lfs/local/0/ranjanr/cc-train/val.bin \
	--output-dir /lfs/local/0/ranjanr/cc-train \
	--vocab-size 50257 \
	--context-length 512 \
	--d-model 768 \
	--num-layers 12 \
	--num-heads 12 \
	--d-ff 3072 \
	--attn-pdrop 0.1 \
	--residual-pdrop 0.1 \
	--batch-size 16 \
	--train-steps 200000 \
	--eval-iters 1000 \
	--eval-interval 2000 \
	--learning-rate 1e-3 \
	--lr-scheduler cosine \
	--weight-decay 0.1 \
	--warmup-ratio 0.01 \
	--grad-clip 1.0 \
	--dtype bfloat16 \
	--wandb-project cc-train \
	--compile

#!/bin/bash

for seed in 0 1 2 3 4
do
	torchrun \
		--standalone \
		--nnodes 1 \
		--nproc_per_node 4 \
		train.py \
		--train-path /lfs/local/0/ranjanr/cc-train/train.bin \
		--dev-path /lfs/local/0/ranjanr/cc-train/val.bin \
		--output-dir /lfs/local/0/ranjanr/cc-train/causal_$seed \
		--vocab-size 50257 \
		--context-length 512 \
		--d-model 768 \
		--num-layers 12 \
		--num-heads 12 \
		--d-ff 3072 \
		--attn-pdrop 0.1 \
		--residual-pdrop 0.1 \
		--batch-size 32 \
		--train-steps 200000 \
		--eval-iters 1000 \
		--eval-interval 2000 \
		--learning-rate 1e-3 \
		--lr-scheduler cosine \
		--weight-decay 0.1 \
		--warmup-ratio 0.01 \
		--grad-clip 1.0 \
		--dtype bfloat16 \
		--wandb-project 2024-07-13__anti_causal_lm \
		--compile \
		--seed $seed
done

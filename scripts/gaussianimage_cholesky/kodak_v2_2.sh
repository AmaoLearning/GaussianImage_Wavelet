#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 9000 7000 5000 3000 1000 800 10_000 30_000 50_000 70_000 90_000 100_000 200_000 300_000 400_000
do
    CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
    --data_name kodak --model_name GaussianImage_Cholesky_v2_2 --num_points $num_points --iterations 50000
done

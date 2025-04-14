#!/bin/bash

# 设置显存阈值，例如100MB
THRESHOLD=8192 #4096

while true; do
    USED_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "当前显存使用量：$USED_MEMORY MB"
    if [ "$USED_MEMORY" -lt "$THRESHOLD" ]; then
        echo "显存空闲，启动程序..."
        python train_dec.py
        break
    else
        echo "显存仍在使用中，等待中..."
    fi
    sleep 600
done
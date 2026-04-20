#!/usr/bin/env bash
set -e

CONFIG_PATH=${1:-configs/train.yaml}

# python scripts/preprocess_data.py --config "$CONFIG_PATH"
python scripts/train.py --config "$CONFIG_PATH"
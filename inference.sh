#!/usr/bin/env bash
set -e

CONFIG_PATH=${1:-configs/inference.yaml}
MESSAGE=${2:-"my card was swallowed by the ATM"}

python scripts/inference.py --config "$CONFIG_PATH" --message "$MESSAGE"
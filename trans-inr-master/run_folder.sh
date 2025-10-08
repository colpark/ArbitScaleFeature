#!/bin/bash

CFG_DIR="./cfgs/imgrec_mamba_composers_ablations"  # path to your folder with config files
echo "Running all cfgs in folder:$1"

for cfg in "./cfgs/$1"/*; do
    if [[ -f "$cfg" ]]; then
        echo "Running config: $cfg"
        yes | python run_trainer.py --cfg "$cfg"
    fi
done
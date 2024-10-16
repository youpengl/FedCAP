#!/bin/bash
device="cuda"
device_id=0
seed=0
dataset="cifar10"
algorithms=("FedCAP" "FedCAP1" "FedCAP2" "FedCAP3" "FedCAP4")
attack_types=("A5" "A6" "A7" "A8")
attack_ratio=0.3

for algorithm in "${algorithms[@]}"; do
    for attack_type in "${attack_types[@]}"; do
        python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio --alpha 10 --phi 0.3 --lamda 0.1
    done
done

attack_ratio=0.0
attack_type="B"
for algorithm in "${algorithms[@]}"; do
    python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio --alpha 10 --phi 0.3 --lamda 0.1
done
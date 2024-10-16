#!/bin/bash
# AGRs w/ Non-IID Denfenses
device="cuda"
device_id=0
seed=0
datasets=("cifar10" "emnist" "wisdm")
AGRs=("mkrum" "median" "rfa")
attack_types=("A5" "A8")
attack_ratio=0.3
gas_p=1000
bucket_s=2

algorithm="FedAvg"
for dataset in "${datasets[@]}"; do
    for attack_type in "${attack_types[@]}"; do
        for agr in "${AGRs[@]}"; do
            python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio --aggregation $agr --gas --gas_p $gas_p
            python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio --aggregation $agr --bucket --bucket_s $bucket_s
            python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio --aggregation $agr
        done  
    done
done

# FedCAP
## A Little is Enough (LIE) Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 10 --phi 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 1 --phi 0.2 --lamda 0.1

## Inner Product Manipulation (IPM) Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 2 --phi 0.1 --lamda 0.1
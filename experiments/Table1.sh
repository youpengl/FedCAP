#!/bin/bash
# Robust Aggregation Methods (AGRs)
device="cuda"
device_id=0
seed=0
datasets=("cifar10" "emnist" "wisdm")
AGRs=("mean" "mkrum" "median" "rfa" "trim" "cluster")
attack_types=("A4" "A3" "A5" "A6" "A7" "A8")
attack_ratio=0.3

algorithm="FedAvg"
for attack_type in "${attack_types[@]}"; do
    for dataset in "${datasets[@]}"; do
        for agr in "${AGRs[@]}"; do
            python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio --aggregation $agr
        done
    done
done

# FLTrust
algorithm="FLTrust"
for attack_type in "${attack_types[@]}"; do
    for dataset in "${datasets[@]}"; do
        python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio
    done
done

# FedCAP
## Sign Flipping (SF) Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A4 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A4 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A4 --attack_ratio 0.3 --alpha 2 --phi 0.2 --lamda 0.1

## Model Replacement (MR) Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A3 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 10.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A3 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A3 --attack_ratio 0.3 --alpha 2 --phi 0.1 --lamda 0.1

## A Little is Enough (LIE) Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 10 --phi 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 1 --phi 0.2 --lamda 0.1

## Min-Max Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A6 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A6 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A6 --attack_ratio 0.3 --alpha 2 --phi 0.2 --lamda 0.5

## Min-Sum Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A7 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A7 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A7 --attack_ratio 0.3 --alpha 2 --phi 0.3 --lamda 0.5

## Inner Product Manipulation (IPM) Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 2 --phi 0.1 --lamda 0.1
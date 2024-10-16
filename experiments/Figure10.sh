#!/bin/bash
# FedAvg & FLTrust
device="cuda"
device_id=0
seed=0
datasets=("cifar10" "emnist")
AGRs=("mkrum" "median" "rfa" "trim" "cluster")
attack_types=("A5" "A6" "A7" "A8")
attack_ratios=(0.1 0.2 0.3 0.4)

algorithm="FedAvg"
for attack_type in "${attack_types[@]}"; do
    for dataset in "${datasets[@]}"; do
        for agr in "${AGRs[@]}"; do
            for attack_ratio in "${attack_ratios[@]}"; do
                python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio --aggregation $agr
            done
        done
    done
done


algorithm="FLTrust"
for attack_type in "${attack_types[@]}"; do
    for dataset in "${datasets[@]}"; do
        for attack_ratio in "${attack_ratios[@]}"; do
            python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio
        done
    done
done

# Ditto & FedCAP
## A Little is Enough (LIE) Attack
### CIFAR10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A5 --attack_ratio 0.1 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A5 --attack_ratio 0.2 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A5 --attack_ratio 0.3 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A5 --attack_ratio 0.4 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A5 --attack_ratio 0.1 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A5 --attack_ratio 0.2 --alpha 10 --phi 0.1 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 10 --phi 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A5 --attack_ratio 0.4 --alpha 10 --phi 0.2 --lamda 0.1

### EMNIST
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A5 --attack_ratio 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A5 --attack_ratio 0.2 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A5 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A5 --attack_ratio 0.4 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A5 --attack_ratio 0.1 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A5 --attack_ratio 0.2 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A5 --attack_ratio 0.4 --alpha 10 --phi 0.2 --lamda 0.5

## Min-Max Attack
### CIFAR10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A6 --attack_ratio 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A6 --attack_ratio 0.2 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A6 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A6 --attack_ratio 0.4 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A6 --attack_ratio 0.1 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A6 --attack_ratio 0.2 --alpha 10 --phi 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A6 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A6 --attack_ratio 0.4 --alpha 10 --phi 0.3 --lamda 0.1

### EMNIST
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A6 --attack_ratio 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A6 --attack_ratio 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A6 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A6 --attack_ratio 0.4 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A6 --attack_ratio 0.1 --alpha 10 --phi 0.2 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A6 --attack_ratio 0.2 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A6 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A6 --attack_ratio 0.4 --alpha 10 --phi 0.1 --lamda 0.5

## Min-Sum Attack
### CIFAR10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A7 --attack_ratio 0.1 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A7 --attack_ratio 0.2 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A7 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A7 --attack_ratio 0.4 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A7 --attack_ratio 0.1 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A7 --attack_ratio 0.2 --alpha 10 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A7 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A7 --attack_ratio 0.4 --alpha 10 --phi 0.3 --lamda 0.5

### EMNIST
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A7 --attack_ratio 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A7 --attack_ratio 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A7 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A7 --attack_ratio 0.4 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A7 --attack_ratio 0.1 --alpha 10 --phi 0.1 --lamda 1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A7 --attack_ratio 0.2 --alpha 10 --phi 0.1 --lamda 1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A7 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A7 --attack_ratio 0.4 --alpha 10 --phi 0.1 --lamda 0.5

## Inner Product Manipulation (IPM) Attack
### CIFAR10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A8 --attack_ratio 0.1 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A8 --attack_ratio 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A8 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A8 --attack_ratio 0.4 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.1 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.2 --alpha 10 --phi 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.4 --alpha 10 --phi 0.1 --lamda 0.5

### EMNIST
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A8 --attack_ratio 0.1 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A8 --attack_ratio 0.2 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A8 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A8 --attack_ratio 0.4 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.1 --alpha 10 --phi 0.2 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.2 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.4 --alpha 10 --phi 0.1 --lamda 0.5
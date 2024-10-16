
#!/bin/bash

# FedAvg & FLTrust & FedROD
device="cuda"
device_id=0
seed=0
datasets=("cifar10" "emnist" "wisdm")
algorithms=("FedAvg" "FedROD")
attack_types=("B" "A1" "A3" "A4")

for attack_type in "${attack_types[@]}"; do
    if [ "$attack_type" = "B" ]; then
        attack_ratio=0.0
    else
        attack_ratio=0.3
    fi
    for algorithm in "${algorithms[@]}"; do
        for dataset in "${datasets[@]}"; do
            python3 system/main.py --device $device --device_id $device_id --seed $seed --dataset $dataset --algorithm $algorithm --attack_type $attack_type --attack_ratio $attack_ratio
        done
    done
done

# Others
## Benign
### Local
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Local --attack_type B --attack_ratio 0.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Local --attack_type B --attack_ratio 0.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm Local --attack_type B --attack_ratio 0.0
### Ditto
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type B --attack_ratio 0.0 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type B --attack_ratio 0.0 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm Ditto --attack_type B --attack_ratio 0.0 --lamda 0.1
### FedFomo
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedFomo --attack_type B --attack_ratio 0.0 --O 10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedFomo --attack_type B --attack_ratio 0.0 --O 10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedFomo --attack_type B --attack_ratio 0.0 --O 18
### FedCAP
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 10 --phi 0.3 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 2 --phi 0.1 --lamda 0.1

## Label Flipping (LF) Attack
# Ditto
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A1 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A1 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm Ditto --attack_type A1 --attack_ratio 0.3 --lamda 0.1
# FedFomo
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedFomo --attack_type A1 --attack_ratio 0.3 --O 10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedFomo --attack_type A1 --attack_ratio 0.3 --O 10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedFomo --attack_type A1 --attack_ratio 0.3 --O 18
# FedCAP
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A1 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A1 --attack_ratio 0.3 --alpha 10 --phi 0.2 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A1 --attack_ratio 0.3 --alpha 4 --phi 0.2 --lamda 0.1

## Sign Flipping (SF) Attack
# Ditto
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A4 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A4 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm Ditto --attack_type A4 --attack_ratio 0.3 --lamda 0.01
# FedFomo
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedFomo --attack_type A4 --attack_ratio 0.3 --O 10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedFomo --attack_type A4 --attack_ratio 0.3 --O 10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedFomo --attack_type A4 --attack_ratio 0.3 --O 18
# FedCAP
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A4 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A4 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A4 --attack_ratio 0.3 --alpha 2 --phi 0.2 --lamda 0.1

## Model Replacement (MR) Attack
# Ditto
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A3 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A3 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm Ditto --attack_type A3 --attack_ratio 0.3 --lamda 0.01
# FedFomo
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedFomo --attack_type A3 --attack_ratio 0.3 --O 10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedFomo --attack_type A3 --attack_ratio 0.3 --O 10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedFomo --attack_type A3 --attack_ratio 0.3 --O 18
# FedCAP
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A3 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 10.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A3 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A3 --attack_ratio 0.3 --alpha 2 --phi 0.1 --lamda 0.1
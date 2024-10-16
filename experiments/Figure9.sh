## CIFAR10
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 2 --phi 0.3 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 5 --phi 0.3 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 10 --phi 0.3 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 2 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 5 --phi 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1
## EMNIST
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 2 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 5 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 10 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 2 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 5 --phi 0.1 --lamda 0.5
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
## WISDM
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 1 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 2 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 3 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type B --attack_ratio 0.0 --alpha 4 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 1 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 2 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 3 --phi 0.1 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset wisdm --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 4 --phi 0.1 --lamda 0.1
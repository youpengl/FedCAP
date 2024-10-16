## A Little is Enough (LIE) Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A5 --attack_ratio 0.3 --lamda 1.0
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A5 --attack_ratio 0.3 --lamda 0.5 --aggregation median
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A5 --attack_ratio 0.3 --lamda 0.5 --aggregation cluster
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A5 --attack_ratio 0.3 --lamda 0.1 --aggregation trim
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 10 --phi 0.2 --lamda 0.1

python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A5 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A5 --attack_ratio 0.3 --lamda 0.1 --aggregation median
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A5 --attack_ratio 0.3 --lamda 0.1 --aggregation cluster
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A5 --attack_ratio 0.3 --lamda 0.1 --aggregation trim
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A5 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5

## Min-Max Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A6 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A6 --attack_ratio 0.3 --lamda 0.1 --aggregation median
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A6 --attack_ratio 0.3 --lamda 1 --aggregation cluster
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A6 --attack_ratio 0.3 --lamda 0.1 --aggregation trim
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A6 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1

python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A6 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A6 --attack_ratio 0.3 --lamda 0.1 --aggregation median
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A6 --attack_ratio 0.3 --lamda 0.1 --aggregation cluster
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A6 --attack_ratio 0.3 --lamda 0.1 --aggregation trim
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A6 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.5

## Min-Sum Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A7 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A7 --attack_ratio 0.3 --lamda 0.1 --aggregation median
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A7 --attack_ratio 0.3 --lamda 0.1 --aggregation cluster
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A7 --attack_ratio 0.3 --lamda 0.1 --aggregation trim
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A7 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1

python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A7 --attack_ratio 0.3 --lamda 0.1
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A7 --attack_ratio 0.3 --lamda 0.1 --aggregation median
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A7 --attack_ratio 0.3 --lamda 0.1 --aggregation cluster
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A7 --attack_ratio 0.3 --lamda 0.1 --aggregation trim
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A7 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5

## Inner Product Manipulation (IPM) Attack
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm Ditto --attack_type A8 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A8 --attack_ratio 0.3 --lamda 0.1 --aggregation median
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A8 --attack_ratio 0.3 --lamda 0.1 --aggregation cluster
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm DittoAGRs --attack_type A8 --attack_ratio 0.3 --lamda 0.1 --aggregation trim
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset cifar10 --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.3 --lamda 0.1

python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm Ditto --attack_type A8 --attack_ratio 0.3 --lamda 0.01
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A8 --attack_ratio 0.3 --lamda 0.1 --aggregation median
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A8 --attack_ratio 0.3 --lamda 0.5 --aggregation cluster
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm DittoAGRs --attack_type A8 --attack_ratio 0.3 --lamda 0.1 --aggregation trim
python3 system/main.py --device cuda --device_id 0 --seed 0 --dataset emnist --algorithm FedCAP --attack_type A8 --attack_ratio 0.3 --alpha 10 --phi 0.1 --lamda 0.5
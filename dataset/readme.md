For convenience, the `FedCAP/dataset/Processed` folder already contains the partitioned user data, and you can directly run the scripts in the `FedCAP/experiments` folder to perform the experiments. 

Additionally, we have provided the code for data partitioning, and you can see the running commands in `FedCAP/experiments/generate_data.sh`. Specifically, the original data is stored in the `FedCAP/dataset/raw` folder. After running the script, the partitioned user data will be stored in the `FedCAP/dataset/processed` folder. 

Please note that you should run the following commmands under the folder `FedCAP`:
```sh
chmod 775 experiments/generate_data.sh
./experiments/generate_data.sh
```
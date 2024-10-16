# Artifact Evaluation of FedCAP

**Abstract:** This is the artifact for the paper "FedCAP: Robust Federated Learning via Customized Aggregation and Personalization", which has been accepted by ACSAC '24. This repository contains configuration guidelines, code, and data (link). The code is used to reproduce all the results in the experimental section of the paper. All experiments were conducted on Ubuntu 22.04 with an NVIDIA RTX A6000 GPU, and the runtime for each Python script is approximately 30 minutes to 1 hour.

**[01/10/2024] News: To simplify the environment setup process, we have made the Docker image public. You can complete the environment setup and run experiments with just three lines of commands.**

```sh
docker pull youpengl/fedcap:acsac2024
docker run -it youpengl/fedcap:acsac2024
./experiments/Figure7.sh
...
```

## 5-Steps Instruction

### 1. Install dependencies

```sh
git clone https://github.com/youpengl/FedCAP.git
cd FedCAP
conda env create -f environment.yml
conda activate 
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip uninstall utils
```

### 2. Download datasets

First, download dataset.zip in [Google Drive](https://drive.google.com/file/d/18ITcYnXXM1veo51D-TqFq2Mv2yExM-HL/view?usp=sharing)/[Dropbox](https://www.dropbox.com/scl/fi/eoxzx6457lgjoms7yoizi/dataset.zip?rlkey=d3476x8wbs3f7zhm3wvgq0mct&st=yz6yrrz1&dl=0) and unzip it.

Then, move the `dataset/*` to the `FedCAP/dataset` folder.

More details about data partitioning can be found in `dataset/readme.md`.

### 3. Run experiments

We provide shell scripts to run experiments for each figure or table in the experimental section.
```
FedCAP/experiments
├── Figure7.sh
├── Figure8.sh
├── Figure9.sh
├── Figure10.sh
├── Figure11.sh
├── Figure12.sh
├── Table1.sh
├── Table2.sh
├── Table3.sh
├── Table4.sh
└── run_all.sh
```
For example, in order to run experiments in Figure 7, you can refer to the following commands:
```sh
chmod 775 experiments/Figure7.sh
./experiments/Figure7.sh
```
You can also refer to the following commands to conduct all experiments at once. Please note that running all the experiments is a time-consuming task.

```sh
chmod 775 experiments/run_all.sh
./experiments/run_all.sh
```

### 4. Evaluation results

After each Python script is executed, the results will be saved in `.npz` files in the `FedCAP/results` folder, and model checkpoints will be saved in the `FedCAP/models` folder. For example, the result file `cifar10_pat_FedCAP_B_0.0_bz10_lr0.01_gr5_ep1_jr1.0_nc20_seed0_lamda0.1_alpha10_phi0.3_normT10.npz` contains `test_acc_g`, `test_acc_p`, `test_accs_g`, `test_accs_p`, `train_loss_g`, and `train_loss_p`, which represent each round's average test accuracy of the global model/customized models, the test accuracy of each user’s global model/customized model, the average test accuracy of personalized models, the test accuracy of each user’s personalized model, and the average training loss of the global model/customized models, as well as the average training loss of personalized models, respectively.

Additionally, the specific meaning of the above .npz file name is as follows:
```
-cifar10: dataset
-pat: pathological non-IID setting (group for EMNIST and nature for WISDM)
-FedCAP: FL method
-B: benign scenarios (attack scenarios include A1: Label Flipping (LF), A3: Model Replacement (MR), A4: Sign Flipping (SF), A5: A Little is Enough (LIE), A6: Min-Max, A7: Min-Sum, and A8: Inner Product Manipulation (IPM) attacks.
-0.0: proportion of malicious clients = 0.0
-bz10: batch_size = 10
-lr0.01: learning_rate = 0.01
-gr5: global_rounds = 5
-ep1: local_epochs = 1
-jr1.0: client parcipanting ratio per round = 1.0
-nc20: the number of all clients = 20
-seed0: seed = 0
-lamda0.1: regularization factor = 0.1 (hyperparameter)
-alpha10: scale factor = 10 (hyperparameter)
-phi0.3: weight factor = 0.3 (hyperparameter)
-normT10: the threshold of anomaly detection = 10 (same in all experiments)
```
### 5. Plot

To accurately reproduce the figures in the paper, we also provide the results and plotting code. Please refer to the following commands for execution:
```sh
cd plot
chmod 775 plot.sh
./plot.sh
```
## Code Tree

```
FedCAP/
├── .gitignore
├── config/ -> Set sweep param. if using Weights & Bias
├── dataset/ -> Used for splitting user data 
│   ├── generate_cifar10.py
│   ├── generate_emnist.py
│   ├── generate_wisdm.py
│   ├── processed/ -> Including partitioned user data
│   ├── raw/ -> Including raw data before splitting
│   ├── readme.md/ Details about data
│   └── utils/ -> Related functions for data generation
├── experiments/ -> Including scripts for experiments
├── readme.md
├── requirements.txt 
├── system/
│   ├── flcore/
│   │   ├── clients/ -> Client classes of FL algorithms
│   │   ├── optimizers/ -> Optimizers of FL algorithms
│   │   ├── servers/ -> Server classes of FL algorithms
│   │   └── trainmodel/ -> Training model structures
│   ├── main.py
│   └── utils/
│       ├── byzantine.py -> Byzantine attack methods
│       ├── config_utils.py -> Parameters settings
│       ├── data_utils.py -> Functions for loading data
│       ├── defense.py -> Robust defense methods
└──     └ tools.py -> Functions for calculation
```


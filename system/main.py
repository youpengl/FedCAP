#!/usr/bin/env python
import copy
import torch
import os
import warnings
import numpy as np
import pprint
import torch.backends.cudnn as cudnn
import random

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serverlocal import Local
from flcore.servers.serverditto import Ditto
from flcore.servers.serverdittoAGRs import DittoAGRs
from flcore.servers.serverrod import FedROD
from flcore.servers.servercap import FedCAP
from flcore.servers.servercap1 import FedCAP1
from flcore.servers.servercap2 import FedCAP2
from flcore.servers.servercap3 import FedCAP3
from flcore.servers.servercap4 import FedCAP4
from flcore.servers.serverfltrust import FLTrust

from flcore.trainmodel.models import *
from utils.config_utils import argparser
from omegaconf import OmegaConf


def run(args):

    print("Creating server and clients ...")

    if args.dataset == 'emnist':
        args.data_path = "dataset/processed/emnist"
        args.partition = 'group'
        args.num_classes = 62
        args.join_ratio = 0.2
        args.num_clients = 100
        args.model = EMNISTCNN(in_features=1, num_classes=args.num_classes, dim=512).to(args.device)

    elif args.dataset == 'cifar10':
        args.data_path = "dataset/processed/cifar10"
        args.partition = 'pat'
        args.num_classes = 10
        args.join_ratio = 1.0
        args.num_clients = 20
        args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

    elif args.dataset == 'wisdm':
        args.data_path = "dataset/processed/wisdm"
        args.partition = 'nature'
        args.num_classes = 6
        args.join_ratio = 1.0
        args.num_clients = 36
        args.model = HARCNN(in_channels=3, num_classes=args.num_classes, dim=3008).to(args.device)
    
    else:
        raise NotImplementedError

    # GPU memory
    print(f"gpu used {torch.cuda.max_memory_allocated(device=None)/1024/1024}MB memory")
    print(args.model)

    # select algorithm
    if args.algorithm == "FedAvg":
        server = FedAvg(args)

    elif args.algorithm == "FLTrust":
        server = FLTrust(args)

    elif args.algorithm == "Local":
        server = Local(args)

    elif args.algorithm == "FedFomo":
        server = FedFomo(args)
    
    elif args.algorithm == "Ditto":
        server = Ditto(args)

    elif args.algorithm == "DittoAGRs":
        server = DittoAGRs(args)

    elif args.algorithm == "FedROD":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedROD(args)

    elif args.algorithm == "FedCAP":
        server = FedCAP(args)

    elif args.algorithm == "FedCAP1":
        server = FedCAP1(args)

    elif args.algorithm == "FedCAP2":
        server = FedCAP2(args)

    elif args.algorithm == "FedCAP3":
        server = FedCAP3(args)

    elif args.algorithm == "FedCAP4":
        server = FedCAP4(args)

    else:
        raise NotImplementedError

    server.train()

    server.save_results()

def main():
    warnings.simplefilter("ignore")
    args = argparser()

    if args.mode == 'single_wandb': # This mode is for Weights & Bias monitor (e.g., monitoring model training processes), please set your parameter setep in system/utils/config_utils.py.
        wandb.init(project=args.project)
        wandb.config.update(vars(args))
        wandb.run.name = eval(args.tag)

    elif args.mode == 'sweep_wandb': # This model is for Weights & Bias sweep (e.g., hyperparameters tuning/parameter sensitive analysis), please set your parameter setep in config/[method]/config.yaml (e.g., config/FedCAP/config.yaml).
        wandb.init(project=args.project)
        wandb.run.name = eval(args.tag)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print('Use device: {}'.format(args.device_id))

    # seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    run(args)


if __name__ == "__main__":
    args = argparser()
    if "wandb" in args.mode:
        import wandb
        os.environ["WANDB_API_KEY"] = "" # if you use wandb, please set your api key that can be found at your Weight & Bias account.
        wandb.login()
    if args.mode == 'sweep_wandb':
        sweep_config = OmegaConf.load("config/{}/sweep_config.yaml".format(args.algorithm))
        pprint.pprint(sweep_config)
        sweep_id = wandb.sweep(OmegaConf.to_container(sweep_config, resolve=True), project=args.project)
        wandb.agent(sweep_id, function=main, project=args.project)

    else:
        main()
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='exp', help="exp")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)

    # Used for Weights & Bias (wandb)
    parser.add_argument('--project', type=str, default="ACSAC", help='project name for wandb')
    parser.add_argument('--tag', type=str, default=None, help='tag name for wandb')
    parser.add_argument('--run_id', type=int, default=None, help='just for tracking on wandb')

    # FL paramters
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--partition', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default=None)
    parser.add_argument('--aggregation', type=str, default="mean")
    parser.add_argument('--attack_type', type=str, default=None)
    parser.add_argument('--attack_ratio', type=float, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--local_learning_rate', type=float, default=0.01)
    parser.add_argument('--learning_rate_decay', type=bool, default=False)
    parser.add_argument('--learning_rate_decay_gamma', type=float, default=0.99)
    parser.add_argument('--global_rounds', type=int, default=100)
    parser.add_argument('--local_steps', type=int, default=5)
    parser.add_argument('--join_ratio', type=float, default=None)
    parser.add_argument('--num_clients', type=int, default=None)
    parser.add_argument('--eval_gap', type=int, default=1)
    parser.add_argument('--detailed_info', type=bool, default=False)

    '''hyperparameters'''
    # Non-IID defense methods
    parser.add_argument('--gas', action='store_true')
    parser.add_argument('--gas_p', type=int, default=None)
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--bucket_s', type=int, default=None)

    # FedFomo
    parser.add_argument('--O', type=int, default=None) 

    # FedCAP
    parser.add_argument('--normT', type=int, default=10)
    parser.add_argument('--alpha_name', type=str, default="")
    parser.add_argument('--alpha', type=int, default=None)
    parser.add_argument('--phi', type=float, default=None)

    # FedCAP & Ditto
    parser.add_argument('--lamda', type=float, default=None)

    args = parser.parse_args()
    return args


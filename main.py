import os
from argparse import ArgumentParser

import numpy as np
import torch

from models import MSHPNet
from myutils.dataset import HRRPDataset
from myutils.graph_matrix import generate_distance_matrix
from myutils.save_and_load_model import count_parameters


def resolve_data_directories(args):
    split_paths = {
        "train": args.train_dir,
        "val": args.val_dir,
        "test": args.test_dir,
    }

    missing_arguments = [f"{split}_dir" for split, path in split_paths.items() if not path]
    if missing_arguments:
        raise ValueError(
            "Missing required dataset directories: "
            + ", ".join(missing_arguments)
            + ". Please provide explicit train/val/test directories."
        )

    resolved_paths = {}
    for split, path in split_paths.items():
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            raise ValueError(f"{split}_dir does not exist or is not a directory: {path}")
        resolved_paths[split] = abs_path

    return resolved_paths


def get_arguments():
    parser = ArgumentParser(description="MSHPNet training")

    parser.add_argument("--epoch_num", type=int, default=100, help="number of training epochs")
    parser.add_argument("--start_epoch", type=int, default=0, help="epoch index to start or resume from")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--seed", type=int, default=56, help="random seed")
    parser.add_argument(
        "--region_loss_weight",
        type=float,
        default=0.5,
        help="weight for the region-aware loss term",
    )

    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint used to resume training")
    parser.add_argument("--save_dir", type=str, default="./model_save", help="directory used to save checkpoints")
    parser.add_argument(
        "--log_tensorboard_dir",
        type=str,
        default="./log/tensorboard",
        help="directory used to save TensorBoard logs",
    )
    parser.add_argument("--log_txt_dir", type=str, default="./log/save_para.txt", help="log file path")
    parser.add_argument("--train_dir", type=str, default=None, help="path to the training split directory")
    parser.add_argument("--val_dir", type=str, default=None, help="path to the validation split directory")
    parser.add_argument("--test_dir", type=str, default=None, help="path to the test split directory")

    parser.add_argument("--HRRP_N", type=int, default=256, help="number of HRRP fast-time bins")
    parser.add_argument("--HRRP_scale", type=int, default=4, help="number of multiscale stages")
    parser.add_argument("--emb_dim", type=int, default=8, help="base embedding dimension")
    parser.add_argument("--gate_hidden_dim", type=int, default=16, help="hidden dimension of the HCSA gating module")
    parser.add_argument("--freq_hidden_dim", type=int, default=64, help="hidden dimension of the frequency-aware block")
    parser.add_argument("--block_num", type=int, default=1, help="number of hybrid perception blocks per stage")

    return parser.parse_args()


def main():
    from train import trainAndTest_model

    args = get_arguments()
    split_dirs = resolve_data_directories(args)

    seed_n = args.seed
    torch.manual_seed(seed_n)
    np.random.seed(seed_n)
    torch.cuda.manual_seed(seed_n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = HRRPDataset(mat_dir=split_dirs["train"], len=args.HRRP_N, data_aug=True)
    val_dataset = HRRPDataset(mat_dir=split_dirs["val"], len=args.HRRP_N)
    test_dataset = HRRPDataset(mat_dir=split_dirs["test"], len=args.HRRP_N)

    n = args.HRRP_N
    scale = args.HRRP_scale
    distance_matrix = []
    for i in range(scale):
        distance_matrix.append(generate_distance_matrix(int(n / (2 ** i))).to(device))

    num_classes = len(train_dataset.labels)
    model = MSHPNet(
        num_classes=num_classes,
        emb_dim=args.emb_dim,
        N=n,
        scale=scale,
        gate_hidden=args.gate_hidden_dim,
        freq_hidden=args.freq_hidden_dim,
        block_num=args.block_num,
    )

    count_parameters(model)
    model.to(device)

    trainAndTest_model(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        distance_matrix,
        args,
        device=device,
    )


if __name__ == "__main__":
    main()

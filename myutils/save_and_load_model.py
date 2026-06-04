import datetime
import os

import torch


def get_checkpoint_path(args):
    return os.path.join(args.save_dir, f"MSHPNet_total_epoch={args.epoch_num}.pth.tar")


def save_checkpoint(model, optimizer, args, epoch_now, best_acc=None):
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch_now,
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "args": args,
        },
        get_checkpoint_path(args),
    )


def reload_best_checkpoint(args, model, best_epoch=None):
    del best_epoch
    path = get_checkpoint_path(args)
    assert os.path.isfile(path), "checkpoint not found."
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(" Evaluating model from best checkpoint")
    return model


def load_training_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path), "resume checkpoint not found."
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(" Resuming training from checkpoint")
    return checkpoint.get("epoch", 0), checkpoint.get("best_acc", float("-inf"))


def print_the_para(file_path, para):
    with open(file_path, "a") as file:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{formatted_time}-----\t{para}\n")


def freeze_and_unfreeze(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    return model


def count_parameters(model):
    result = {}
    total = 0
    for name, module in model.named_children():
        param_sum = sum(p.numel() for p in module.parameters())
        result[name] = param_sum
        total += param_sum
    result["Total"] = total
    for name, count in result.items():
        print(f"{name}: {count / 1e6} M")

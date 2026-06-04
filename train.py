import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from myutils.loss_function import multi_scale_region_aware_loss
from myutils.save_and_load_model import (
    load_training_checkpoint,
    print_the_para,
    reload_best_checkpoint,
    save_checkpoint,
)


def build_label_name_map(labels):
    return {idx: name for name, idx in sorted(labels.items(), key=lambda item: item[1])}


def compute_total_loss(
    logits,
    labels,
    predicted_regions,
    target_regions,
    scale,
    region_loss_weight,
    classification_criterion=None,
    region_loss_fn=multi_scale_region_aware_loss,
):
    if classification_criterion is None:
        classification_criterion = nn.CrossEntropyLoss()

    classification_loss = classification_criterion(logits, labels)
    region_loss = region_loss_fn(predicted_regions, target_regions, scale)
    total_loss = classification_loss + region_loss_weight * region_loss
    return total_loss, classification_loss, region_loss


def evaluate_split(model, data_loader, args, device, classification_criterion, collect_features=False):
    model.eval()
    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    region_loss_sum = 0.0
    step_count = 0
    all_labels = []
    predicted_labels = []
    features = []
    sample_input = None

    with torch.no_grad():
        for hrrp_one, hrrp_label, label, _filename in data_loader:
            hrrp_one = hrrp_one.to(device).to(torch.float32)
            hrrp_label = hrrp_label.to(device).to(torch.float32)
            label = label.long().to(device)

            outputs, _sub_label, region_predictions, embeddings = model(hrrp_one)
            total_loss, classification_loss, region_loss = compute_total_loss(
                logits=outputs,
                labels=label,
                predicted_regions=region_predictions,
                target_regions=hrrp_label,
                scale=args.HRRP_scale,
                region_loss_weight=args.region_loss_weight,
                classification_criterion=classification_criterion,
            )

            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            total_loss_sum += total_loss.item()
            classification_loss_sum += classification_loss.item()
            region_loss_sum += region_loss.item()
            step_count += 1

            if collect_features:
                features.extend(embeddings.cpu().numpy())
                if sample_input is None:
                    sample_input = hrrp_one[:1].detach().cpu()

    accuracy = 0.0
    if all_labels:
        accuracy = 100 * np.mean(np.array(predicted_labels) == np.array(all_labels))

    if step_count == 0:
        return {
            "total_loss": 0.0,
            "classification_loss": 0.0,
            "region_loss": 0.0,
            "accuracy": accuracy,
            "all_labels": all_labels,
            "predicted_labels": predicted_labels,
            "features": features,
            "sample_input": sample_input,
        }

    return {
        "total_loss": total_loss_sum / step_count,
        "classification_loss": classification_loss_sum / step_count,
        "region_loss": region_loss_sum / step_count,
        "accuracy": accuracy,
        "all_labels": all_labels,
        "predicted_labels": predicted_labels,
        "features": features,
        "sample_input": sample_input,
    }


def _maybe_profile_model(model, sample_input, device):
    if sample_input is None:
        return

    try:
        from thop import profile
    except Exception as exc:
        warnings.warn(f"Skipping FLOPs profiling because THOP is unavailable: {exc}")
        return

    sample_input = sample_input.to(device)
    flops, params = profile(model, inputs=(sample_input,))
    print(f"FLOPs: {flops / 1e9} GFLOPs")
    print(f"Params: {params / 1e6} M")
    start = time.time()
    _ = model(sample_input)
    end = time.time()
    print(f"Elapsed time: {end - start} seconds")


def _maybe_draw_reports(features, all_labels, label_name_map, predicted_labels):
    try:
        from myutils.confusion_matrix import draw_confusion_matrix, draw_tsne
    except Exception as exc:
        warnings.warn(f"Skipping confusion matrix and T-SNE generation: {exc}")
        return

    if features:
        draw_tsne(features, all_labels, label_name_map)
    draw_confusion_matrix(
        label_true=all_labels,
        label_pred=predicted_labels,
        label_name=label_name_map,
    )


def _maybe_compute_classification_metrics(all_labels, predicted_labels):
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
    except Exception as exc:
        warnings.warn(f"Skipping sklearn classification metrics: {exc}")
        return None

    return {
        "f1": f1_score(y_true=all_labels, y_pred=predicted_labels, average="macro"),
        "precision": precision_score(y_true=all_labels, y_pred=predicted_labels, average="macro"),
        "recall": recall_score(y_true=all_labels, y_pred=predicted_labels, average="macro"),
    }


def _log_epoch_summary(writer, epoch, train_metrics, val_metrics):
    writer.add_scalar("train/total_loss", train_metrics["total_loss"], epoch)
    writer.add_scalar("train/classification_loss", train_metrics["classification_loss"], epoch)
    writer.add_scalar("train/region_loss", train_metrics["region_loss"], epoch)
    writer.add_scalar("val/total_loss", val_metrics["total_loss"], epoch)
    writer.add_scalar("val/classification_loss", val_metrics["classification_loss"], epoch)
    writer.add_scalar("val/region_loss", val_metrics["region_loss"], epoch)
    writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)


def trainAndTest_model(model, train_dataset, val_dataset, test_dataset, distance_matrix, args, device):
    del distance_matrix

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.dirname(args.log_txt_dir)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)
    classification_criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(args.log_tensorboard_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 4, shuffle=False)

    start_epoch = args.start_epoch
    best_acc = float("-inf")
    best_epoch = -1

    if args.resume:
        checkpoint_epoch, checkpoint_best_acc = load_training_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint_epoch + 1
        best_acc = checkpoint_best_acc

    for epoch in range(start_epoch, args.epoch_num):
        model.train()
        train_total_loss = 0.0
        train_classification_loss = 0.0
        train_region_loss = 0.0
        step_count = 0
        epoch_start = time.time()

        for hrrp_one, hrrp_label, label, _filename in train_loader:
            optimizer.zero_grad()
            hrrp_one = hrrp_one.to(device).to(torch.float32)
            hrrp_label = hrrp_label.to(device).to(torch.float32)
            label = label.long().to(device)

            outputs, _sub_label, region_predictions, _embeddings = model(hrrp_one)
            total_loss, classification_loss, region_loss = compute_total_loss(
                logits=outputs,
                labels=label,
                predicted_regions=region_predictions,
                target_regions=hrrp_label,
                scale=args.HRRP_scale,
                region_loss_weight=args.region_loss_weight,
                classification_criterion=classification_criterion,
            )

            total_loss.backward()
            optimizer.step()

            train_total_loss += total_loss.item()
            train_classification_loss += classification_loss.item()
            train_region_loss += region_loss.item()
            step_count += 1

        scheduler.step()
        elapsed = time.time() - epoch_start

        train_metrics = {
            "total_loss": train_total_loss / max(step_count, 1),
            "classification_loss": train_classification_loss / max(step_count, 1),
            "region_loss": train_region_loss / max(step_count, 1),
        }
        val_metrics = evaluate_split(
            model=model,
            data_loader=val_loader,
            args=args,
            device=device,
            classification_criterion=classification_criterion,
        )
        _log_epoch_summary(writer, epoch, train_metrics, val_metrics)

        print(
            f"Epoch [{epoch + 1}/{args.epoch_num}] "
            f"train_total_loss={train_metrics['total_loss']:.4f}, "
            f"val_total_loss={val_metrics['total_loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.2f}%, "
            f"runtime={elapsed:.2f}s"
        )

        if val_metrics["accuracy"] >= best_acc:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, args, epoch, best_acc=best_acc)
            print(f"New best validation checkpoint saved at epoch {epoch + 1} with acc {best_acc:.2f}%")

    if best_epoch < 0:
        save_checkpoint(model, optimizer, args, args.epoch_num - 1, best_acc=best_acc)
        best_epoch = args.epoch_num - 1

    model = reload_best_checkpoint(args, model)
    test_metrics = evaluate_split(
        model=model,
        data_loader=test_loader,
        args=args,
        device=device,
        classification_criterion=classification_criterion,
        collect_features=True,
    )

    _maybe_profile_model(model, test_metrics["sample_input"], device)
    label_name_map = build_label_name_map(train_dataset.labels)
    _maybe_draw_reports(
        features=test_metrics["features"],
        all_labels=test_metrics["all_labels"],
        label_name_map=label_name_map,
        predicted_labels=test_metrics["predicted_labels"],
    )

    print(f"Training completed. Best epoch = {best_epoch}")
    print_the_para(args.log_txt_dir, f"Best epoch: {best_epoch}")
    print_the_para(args.log_txt_dir, f"Test total loss: {test_metrics['total_loss']:.6f}")
    print_the_para(args.log_txt_dir, f"Test CE loss: {test_metrics['classification_loss']:.6f}")
    print_the_para(args.log_txt_dir, f"Test region-aware loss: {test_metrics['region_loss']:.6f}")

    print(f"Accuracy on test set: {test_metrics['accuracy']:.2f}%")
    print_the_para(args.log_txt_dir, f"Accuracy on test set: {test_metrics['accuracy']:.2f}%")

    extra_metrics = _maybe_compute_classification_metrics(
        test_metrics["all_labels"],
        test_metrics["predicted_labels"],
    )
    if extra_metrics is not None:
        print(f"F1 Score: {100 * extra_metrics['f1']:.2f}%")
        print(f"Precision: {100 * extra_metrics['precision']:.2f}%")
        print(f"Recall: {100 * extra_metrics['recall']:.2f}%")
        print_the_para(args.log_txt_dir, f"F1 Score: {extra_metrics['f1']}")
        print_the_para(args.log_txt_dir, f"Precision: {extra_metrics['precision']}")
        print_the_para(args.log_txt_dir, f"Recall: {extra_metrics['recall']}")

    label_to_count = {i: 0 for i in range(len(train_dataset.labels))}
    correct_predictions_per_label = {i: 0 for i in range(len(train_dataset.labels))}
    for true_label, pred_label in zip(test_metrics["all_labels"], test_metrics["predicted_labels"]):
        label_to_count[true_label] += 1
        if true_label == pred_label:
            correct_predictions_per_label[true_label] += 1

    class_accuracies = {
        label: (correct / count) * 100
        for label, correct, count in zip(
            correct_predictions_per_label.keys(),
            correct_predictions_per_label.values(),
            label_to_count.values(),
        )
        if count > 0
    }
    for label, accuracy in class_accuracies.items():
        print(f"Class {label}: Accuracy = {accuracy:.2f}%")
        print_the_para(args.log_txt_dir, f"Class {label}: Accuracy = {accuracy:.2f}%")

    writer.close()

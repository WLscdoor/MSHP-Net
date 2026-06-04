# MSHP-Net

PyTorch implementation of **MSHP-Net** for HRRP target recognition.

## Overview

MSHP-Net combines:

- a **Hybrid Perception Encoder**
- a **Lightweight Region Decoder**
- a **Hierarchical Cross-Scale Aggregation block**

The current training pipeline uses:

- classification loss: `CrossEntropyLoss`
- region-aware loss: `multi_scale_region_aware_loss`
- total loss: `CE + beta * region-aware loss`

## Repository Structure

```text
.
|-- main.py
|-- train.py
|-- models.py
`-- myutils/
```

## Requirements

Recommended environment:

- Python 3.10+
- PyTorch
- torch-geometric
- numpy
- scipy
- opencv-python
- torchvision
- pillow
- tensorboard
- info-nce-pytorch or a compatible `info_nce` package

Optional packages:

- `thop` for FLOPs / parameter profiling
- `scikit-learn` for F1 / precision / recall and visualization helpers

## Dataset Layout

Prepare three directories and pass them explicitly:

```text
your_dataset/
|-- train/
|-- val/
`-- test/
```

Each split directory should contain `.mat` files in the format expected by `myutils/dataset.py`.

## Training

Run training with:

```powershell
python main.py `
  --train_dir /path/to/train `
  --val_dir /path/to/val `
  --test_dir /path/to/test
```

On Linux or macOS:

```bash
python main.py \
  --train_dir /path/to/train \
  --val_dir /path/to/val \
  --test_dir /path/to/test
```

Useful arguments:

- `--epoch_num`
- `--batch_size`
- `--learning_rate`
- `--region_loss_weight`
- `--save_dir`
- `--resume`

## Notes

- The dataset label mapping is currently defined in `myutils/dataset.py`.
- TensorBoard logs are written to `./log/tensorboard` by default.
- Best checkpoints are saved under `./model_save`.

## Citation

If you use this repository, please cite the following paper:

```bibtex
@ARTICLE{11145212,
  author={Li, Xiaodi and Hou, Yuguan and Xu, Zihan and Jin, Xinfei and Su, Fulin and Li, Hongxu},
  journal={IEEE Transactions on Radar Systems},
  title={A Multiscale Hybrid Perception Network With Granularity Decoupling and Spectral Enhancement for HRRP Target Recognition},
  year={2025},
  volume={3},
  number={},
  pages={1183-1194},
  keywords={Feature extraction;Target recognition;Scattering;Transformers;Sensitivity;Decoding;Radar imaging;Hidden Markov models;Adaptation models;High-resolution imaging;Synthetic aperture radar;High-resolution range profile (HRRP);inverse synthetic aperture radar (ISAR);radar automatic target recognition (RATR)},
  doi={10.1109/TRS.2025.3604214}
}
```

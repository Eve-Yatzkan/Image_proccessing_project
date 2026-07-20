# Robustness Evaluation of Classical and Deep Computer Vision Methods

Image Processing Course Project

## Overview

This project evaluates the robustness of computer vision methods under controlled image distortions. It uses the PASCAL VOC 2012 `train` split and compares one classical method, ORB feature detection and matching, with two pretrained deep learning models: DeepLabV3-ResNet50 for semantic segmentation and YOLO11n for object detection.

The current implementation includes dataset exploration, clean baseline evaluation, distortion generation, and robustness analysis. Image enhancement is the next planned stage, followed by a comparison of clean, distorted, and enhanced performance.

## Table of Contents

- [Project Requirements](#project-requirements)
- [Dataset](#dataset)
- [Computer Vision Tasks](#computer-vision-tasks)
- [Image Distortions](#image-distortions)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Current Results](#current-results)
- [Visual Results](#visual-results)
- [Repository Structure](#repository-structure)
- [Notebook Guide](#notebook-guide)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Project Status](#project-status)
- [Limitations](#limitations)
- [References](#references)
- [Authors](#authors)

## Project Requirements

The course project requires:

- A public dataset with ground-truth annotations
- Three computer vision tasks
- Classical and deep learning methods
- Three image distortions
- Quantitative evaluation on clean and distorted images
- An enhancement or fine-tuning stage

## Dataset

- Dataset: PASCAL VOC 2012
- Split: `train`
- Loader: `torchvision.datasets.VOCSegmentation`
- Sample size: fixed subset of 100 images
- Random seed: `10`
- Class coverage: all 20 VOC foreground classes are represented
- Ground truth:
  - Semantic segmentation masks
  - VOC XML object-detection annotations
  - Ignore label `255` excluded from segmentation evaluation

Saved dataset artifacts:

- `results/selected_indices.csv`
- `results/selected_image_sizes.csv`
- `results/segmentation_class_summary.csv`
- `results/detection_class_summary.csv`

## Computer Vision Tasks

### ORB Feature Detection and Matching

- Method: OpenCV ORB
- Configuration: `cv2.ORB_create(nfeatures=1500)`
- Matching: Hamming-distance brute-force matching with `knnMatch(..., k=2)`
- Ratio test: `0.75`
- Robustness metric: normalized match ratio between clean and distorted descriptors
- Key outputs: `orb_clean_summary.csv`, `orb_distortion_summary.csv`

### Semantic Segmentation

- Model: DeepLabV3-ResNet50
- Weights: `DeepLabV3_ResNet50_Weights.DEFAULT`
- Evaluation: predicted masks compared with PASCAL VOC masks
- Ignore handling: VOC label `255` is excluded
- Metrics: per-image mIoU, dataset mIoU, and foreground mIoU
- Key outputs: `segmentation_clean_summary.csv`, `segmentation_distortion_summary.csv`

### Object Detection

- Model: Ultralytics `YOLO("yolo11n.pt")`
- Confidence threshold: `0.25`
- IoU threshold: `0.5`
- Evaluation: YOLO predictions compared with VOC XML boxes
- VOC handling: difficult objects ignored; selected COCO class names are mapped to VOC equivalents
- Matching: class-aware greedy one-to-one assignment
- Metrics: precision, recall, F1-score, and matched IoU
- Key outputs: `detection_clean_summary.csv`, `detection_distortion_summary.csv`

## Image Distortions

| Distortion | Mild | Medium | Severe | Implementation |
|---|---:|---:|---:|---|
| Gaussian noise | sigma `10` | sigma `25` | sigma `50` | Adds zero-mean Gaussian RGB noise and clips to `[0, 255]` |
| JPEG compression | quality `50` | quality `20` | quality `5` | Saves and reloads each RGB image through an in-memory JPEG buffer |
| Low light | factor `0.6` | factor `0.35` | factor `0.15` | Multiplies RGB pixel values by the brightness factor |

SNR is computed in dB as `10 * log10(signal_power / error_power)`, using the clean image as the signal and the clean-distorted difference as the error.

The distortion configuration is saved to `results/distortion_config.csv`.

## Evaluation Pipeline

```text
PASCAL VOC 2012
-> Fixed 100-image subset
-> Clean baseline
-> Distortion generation
-> Robustness evaluation
-> Image enhancement and final comparison (planned)
```

## Current Results

### Clean Baseline

| Task | Clean result |
|---|---|
| ORB | Mean keypoints `1336.09`; median `1441`; `18` images reached the 1500-feature cap |
| Semantic segmentation | Dataset foreground mIoU `0.777`; mean per-image foreground mIoU `0.700` |
| Object detection | Global precision `0.651`; recall `0.808`; F1 `0.721`; mean matched IoU `0.871` |

### Main Robustness Findings

- Gaussian noise causes gradual degradation across all tasks. At severe noise, ORB normalized match ratio falls to `0.399`, segmentation foreground mIoU to `0.493`, and detection F1 to `0.313`.
- Severe JPEG compression produces the strongest degradation for segmentation and detection, with foreground mIoU `0.174` and detection F1 `0.145`.
- Low light sharply reduces the absolute number of ORB matches. At severe low light, ORB keeps a normalized match ratio of `0.813`, but mean good matches drop to `62.83`.
- YOLO is comparatively robust to the tested low-light range: detection F1 is `0.729`, `0.714`, and `0.686` for mild, medium, and severe low light.

## Visual Results

![PASCAL VOC segmentation ground truth](figures/dataset_segmentation_ground_truth_grid.png)

![Gaussian noise severity examples](figures/gaussian_noise_severity_examples.png)

![ORB normalized match ratio under distortions](figures/orb_normalized_match_ratio_distortions.png)

![Segmentation foreground IoU under distortions](figures/segmentation_foreground_iou_distortions.png)

![Detection global F1 under distortions](figures/detection_global_f1_distortions.png)

## Repository Structure

```text
.
+-- README.md
+-- 3002_CousreProject.pdf                  # project instructions
+-- requirements.txt                        # Python dependencies
+-- notebooks/
|   +-- 01_dataset_exploration.ipynb
|   +-- 02_clean_baseline.ipynb
|   +-- 03_distortions.ipynb
+-- results/                                # generated CSV summaries and detailed evaluations
+-- figures/                                # generated visual examples and performance plots
```

## Notebook Guide

### `01_dataset_exploration.ipynb`

Loads PASCAL VOC 2012, sets the fixed seed, selects the 100-image subset, verifies class coverage, visualizes segmentation and detection ground truth, and saves dataset metadata to `results/` and `figures/`.

### `02_clean_baseline.ipynb`

Runs clean-image baselines for ORB, DeepLabV3-ResNet50, and YOLO11n. It saves per-image and aggregate metrics for feature detection, segmentation, and object detection.

### `03_distortions.ipynb`

Defines the three distortions, computes SNR, evaluates all three tasks under every distortion-severity combination, and saves robustness summaries and plots.

## Installation

```bash
pip install -r requirements.txt
```

Optional:

```bash
jupyter notebook
```

## Usage

Run the notebooks in order:

```text
01_dataset_exploration.ipynb
02_clean_baseline.ipynb
03_distortions.ipynb
04_image_enhancement.ipynb (planned)
```

The notebooks use repository-relative paths. PASCAL VOC 2012 is downloaded or loaded through `torchvision` according to the notebook configuration. Outputs are saved to `results/` and `figures/`.

## Reproducibility

- Fixed random seed: `10`
- Fixed sample size: 100 images
- Saved sample indices: `results/selected_indices.csv`
- Repository-relative paths for `data/`, `results/`, and `figures/`
- Dependencies installed from `requirements.txt`
- Saved CSV results and figures for completed stages

## Project Status

- [x] Dataset exploration
- [x] Clean baseline evaluation
- [x] Distortion generation
- [x] Robustness evaluation
- [ ] Image enhancement
- [ ] Evaluation after enhancement
- [ ] Fine-tuning, if required
- [ ] Final comparison and reporting

## Limitations

- Evaluation uses a fixed subset of 100 images.
- The deep models are pretrained and have not yet been fine-tuned.
- Image enhancement has not yet been implemented.
- Results are specific to the selected distortions and severity levels.

## References

- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [OpenCV ORB Documentation](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html)
- [Torchvision DeepLabV3 Documentation](https://pytorch.org/vision/stable/models/deeplabv3.html)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)

## Authors

- Eve Yatzkan
- Maayan Zelig

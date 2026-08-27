# Smart Pothole Detection Framework Using Deep Learning Techniques

A deep learning framework for automatically detecting and localizing potholes in road images. The project trains and compares three object detection architectures — **YOLOv8**, **Faster R-CNN** (ResNet50-FPN backbone), and **SSD MobileNet** — on an annotated pothole dataset, covering the full pipeline from raw data to trained, evaluated models.

## Overview

Road surface damage such as potholes poses a safety hazard and is costly to identify manually at scale. This project builds an end-to-end computer vision pipeline that:

- Loads and organizes a real-world pothole image dataset with Pascal VOC XML annotations
- Explores the data through visual and statistical analysis (EDA)
- Preprocesses images for consistent, high-quality model input
- Converts annotations into the formats required by each detection architecture
- Trains and evaluates three object detection models on the same data splits
- Compares the models across detection accuracy, localization quality, and inference speed

## Dataset

[Pothole Detection Dataset – Kaggle](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)

Annotations are provided in Pascal VOC XML format (bounding boxes with `xmin`, `ymin`, `xmax`, `ymax`). The dataset is split into **train (70%)**, **validation (20%)**, and **test (10%)** sets, with a fixed random seed for reproducibility.

## Pipeline

### 1. Dataset Loading & Path Configuration
Images and their corresponding XML annotation files are located, paired, shuffled, and split into train/val/test directories.

### 2. Exploratory Data Analysis (EDA)
Visual analysis of the dataset, including:
- Split distribution (annotation counts per split)
- Bounding box area and aspect ratio distributions
- Spatial heatmap of pothole locations within images
- Image resolution distribution
- Bounding box width vs. height relationships
- Number of annotations per image
- Sample annotated image visualization

### 3. Image Preprocessing
Three preprocessing steps are applied to standardize and enhance input images:
- **Resizing** — all images resized to a fixed target size (640×640) for consistent model input
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — improves local contrast and pothole visibility under varying lighting/road conditions
- **Normalization** — pixel values scaled to a smaller numerical range to stabilize training

### 4. Model Training
All three models are trained on the same preprocessed data splits with shared hyperparameters where applicable (epochs, image size, batch size, confidence/IoU thresholds):

- **YOLOv8** — annotations converted to YOLO format, trained via the `ultralytics` package, with a dataset YAML config generated automatically
- **Faster R-CNN** — ResNet50-FPN backbone from `torchvision`, fine-tuned with a custom `FastRCNNPredictor` head
- **SSD MobileNet** — lightweight single-shot detector trained via `torchvision`

### 5. Evaluation & Model Comparison
Each model is evaluated on the held-out test set using:
- Precision, Recall, F1-score
- Mean IoU (Intersection over Union)
- True Positives / False Positives / False Negatives
- Inference speed per image

Results are visualized through comparative bar charts, radar charts, precision-recall trade-off plots, and a suitability summary table. Actual-vs-predicted bounding box overlays are also generated for qualitative inspection.

## Challenges Addressed

- **Variable pothole size and scale** across different road textures and conditions
- **Small object detection**, which increases the risk of missed detections
- **Annotation parsing**, correctly extracting and mapping Pascal VOC XML bounding boxes while handling image-annotation mismatches
- **Speed vs. accuracy trade-offs** across the three architectures
- **False negative risk**, which is particularly critical in road-safety applications

## Tech Stack

- **Python** (3.12)
- **PyTorch** / **torchvision** — Faster R-CNN, SSD MobileNet
- **Ultralytics YOLO** — YOLOv8
- **OpenCV**, **Pillow** — image processing (resizing, CLAHE, color conversion)
- **NumPy**, **Pandas** — data handling and analysis
- **Matplotlib** — EDA and results visualization
- **tqdm** — progress tracking

## Project Structure

```
pothole_dataset/         # Raw train/val/test images + Pascal VOC XML annotations
pothole_preprocessed/    # Resized, CLAHE-enhanced, normalized images
pothole_yolo/            # YOLO-format images, labels, and dataset YAML config
runs/yolo/                # YOLOv8 training runs and weights
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install ultralytics opencv-python pillow numpy pandas matplotlib tqdm torch torchvision
   ```
2. Download the dataset from Kaggle and update the `INPUT_DIR` path in the notebook.
3. Run the notebook cells in order: dataset preparation → EDA → preprocessing → model training (YOLOv8, Faster R-CNN, SSD MobileNet) → evaluation & comparison.

## Notes

- The notebook was developed and run on Kaggle with GPU acceleration (NVIDIA Tesla T4).
- A random seed (42) is set across `random`, `numpy`, and `torch` for reproducibility.
- The intended end use is a lightweight web application where a user uploads a road image and receives detection results (annotated image, count of damaged areas, and confidence scores) from the trained model.

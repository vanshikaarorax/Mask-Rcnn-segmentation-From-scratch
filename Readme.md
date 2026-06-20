# 🎯 Mask R-CNN: Complete Implementation Guide

## From-Scratch Implementation for Object Detection & Instance Segmentation

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why From Scratch?](#-why-from-scratch)
3. [Complete Architecture](#-complete-architecture)
4. [Component Breakdown](#-component-breakdown)
5. [Training vs Inference Flow](#-training-vs-inference-flow)
6. [Data Flow Pipeline](#-data-flow-pipeline)
7. [Key Features](#-key-features)
8. [Technical Stack](#-technical-stack)
9. [Quick Start](#-quick-start)
10. [Project Structure](#-project-structure)
11. [Learning Outcomes](#-learning-outcomes)
12. [License & Acknowledgments](#-license--acknowledgments)

---

## 🎯 Project Overview

**Mask R-CNN From Scratch** is a complete, production-ready implementation of the state-of-the-art Mask R-CNN architecture built entirely from the ground up using TensorFlow/Keras. Unlike other implementations that import pre-built ResNet or pre-trained models, **every single layer is defined and built from scratch**, giving you complete control and understanding of the entire architecture.

| Aspect | Details |
|--------|---------|
| **Type** | Object Detection + Instance Segmentation |
| **Implementation** | 100% From Scratch (No imported models) |
| **Framework** | TensorFlow 2.14+ / Keras |
| **Backbone** | ResNet50/101 (Built from layers) |
| **Architecture** | ResNet + FPN + RPN + ROI Align + Heads |

---

## 🤔 Why From Scratch?

| Reason | Benefit |
|--------|---------|
| **Complete Control** | Modify any layer without framework limitations |
| **Full Transparency** | Understand every operation in the pipeline |
| **Educational Value** | Learn Mask R-CNN internals by building them |
| **Flexibility** | Add custom layers, losses, or modifications easily |
| **No Black Boxes** | Everything is visible and debuggable |

---

## 🏗️ Complete Architecture

### High-Level Overview
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MASK R-CNN COMPLETE ARCHITECTURE │
│ (Built 100% From Scratch) │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ │
│ INPUT IMAGE [batch, H, W, 3] │
│ │ │
│ ▼ │
│ ┌───────────────────────────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 1: RESNET BACKBONE (resnet_graph) ││
│ │ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ ││
│ │ │ Stage 1: Conv7x7 + MaxPool → C1 [256x256x64] │ ││
│ │ │ Stage 2: Conv + 2x Identity → C2 [256x256x256] ← Fine details │ ││
│ │ │ Stage 3: Conv + 3x Identity → C3 [128x128x512] ← Object parts │ ││
│ │ │ Stage 4: Conv + 5/22x Identity → C4 [64x64x1024] ← Whole objects │ ││
│ │ │ Stage 5: Conv + 2x Identity → C5 [32x32x2048] ← Semantic context │ ││
│ │ └──────────────────────────────────────────────────────────────────────────────────────────┘ ││
│ └───────────────────────────────────────────────────────────────────────────────────────────────┘│
│ │ │
│ ▼ │
│ ┌───────────────────────────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 2: FPN CONSTRUCTION (build() method) ││
│ │ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ ││
│ │ │ C5 → Conv1x1 → P5 [32x32x256] ← Largest objects │ ││
│ │ │ C4 → Conv1x1 + Upsample(P5) → P4 [64x64x256] ← Large objects │ ││
│ │ │ C3 → Conv1x1 + Upsample(P4) → P3 [128x128x256] ← Medium objects │ ││
│ │ │ C2 → Conv1x1 + Upsample(P3) → P2 [256x256x256] ← Small objects │ ││
│ │ │ P5 → MaxPool → P6 [16x16x256] ← Extra large (RPN only) │ ││
│ │ └──────────────────────────────────────────────────────────────────────────────────────────┘ ││
│ └───────────────────────────────────────────────────────────────────────────────────────────────┘│
│ │ │
│ ▼ │
│ ┌───────────────────────────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 3: RPN (rpn_graph) ││
│ │ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ ││
│ │ │ For each P2-P6: │ ││
│ │ │ Shared Conv3x3(512) → Branch │ ││
│ │ │ ├── Classification Head: Conv1x1(2A) → rpn_class_logits [batch, total_anchors, 2] │ ││
│ │ │ └── BBox Head: Conv1x1(4A) → rpn_bbox [batch, total_anchors, 4] │ ││
│ │ └──────────────────────────────────────────────────────────────────────────────────────────┘ ││
│ └───────────────────────────────────────────────────────────────────────────────────────────────┘│
│ │ │
│ ▼ │
│ ┌───────────────────────────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 4: PROPOSAL LAYER (ProposalLayer) ││
│ │ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ ││
│ │ │ 1. Select top 6000 anchors by FG score │ ││
│ │ │ 2. apply_box_deltas_graph() → Refine boxes │ ││
│ │ │ 3. clip_boxes_graph() → Clip to [0,1] │ ││
│ │ │ 4. NMS → Keep top 2000 (training) or 1000 (inference) │ ││
│ │ │ Output: rpn_rois [batch, 2000/1000, 4] │ ││
│ │ └──────────────────────────────────────────────────────────────────────────────────────────┘ ││
│ └───────────────────────────────────────────────────────────────────────────────────────────────┘│
│ │ │ │
│ │ │ │
│ ▼ ▼ │
│ ┌─────────────────────────────────┐ ┌──────────────────────────────────────────────────────────┐│
│ │ TRAINING PATH │ │ INFERENCE PATH ││
│ │ ┌─────────────────────────────┐│ │ ┌────────────────────────────────────────────────────┐ ││
│ │ │ DetectionTargetLayer ││ │ │ Heads (Direct from rpn_rois) │ ││
│ │ │ Inputs: rpn_rois + GT ││ │ └────────────────────────────────────────────────────┘ ││
│ │ │ 1. IoU Computation ││ │ │ ││
│ │ │ 2. Classify proposals ││ │ ▼ ││
│ │ │ 3. Subsample to 200 ││ │ ┌────────────────────────────────────────────────────┐ ││
│ │ │ 4. Generate targets ││ │ │ DetectionLayer │ ││
│ │ │ Outputs: ││ │ │ 1. Get best class (argmax) │ ││
│ │ │ - rois [200,4] ││ │ │ 2. Class-specific deltas │ ││
│ │ │ - target_class_ids [200] ││ │ │ 3. Refine boxes │ ││
│ │ │ - target_bbox [200,4] ││ │ │ 4. Filter background + confidence │ ││
│ │ │ - target_mask [200,28,28]││ │ │ 5. Class-wise NMS │ ││
│ │ └─────────────────────────────┘│ │ │ Output: detections [max, 6] │ ││
│ │ │ │ │ └────────────────────────────────────────────────────┘ ││
│ │ ▼ │ │ │ ││
│ │ ┌─────────────────────────────┐│ │ ▼ ││
│ │ │ PyramidROIAlign (7x7/14x14)││ │ ┌────────────────────────────────────────────────────┐ ││
│ │ │ 1. Assign to FPN level ││ │ │ build_fpn_mask_graph() │ ││
│ │ │ 2. Crop features ││ │ │ detection_boxes + P2-P5 → masks │ ││
│ │ │ 3. Resize to fixed size ││ │ │ Output: mrcnn_mask [max, 28, 28, C] │ ││
│ │ └─────────────────────────────┘│ │ └────────────────────────────────────────────────────┘ ││
│ │ │ │ │ │ ││
│ │ ▼ │ │ ▼ ││
│ │ ┌─────────────────────────────┐│ │ ┌────────────────────────────────────────────────────┐ ││
│ │ │ fpn_classifier_graph() ││ │ │ FINAL OUTPUT │ ││
│ │ │ ROI features (7x7) → ││ │ │ detections + mrcnn_mask │ ││
│ │ │ 2x FC Layers → ││ │ └────────────────────────────────────────────────────┘ ││
│ │ │ ├── Classifier Head (C) ││ │ ││
│ │ │ └── BBox Head (C,4) ││ │ ││
│ │ └─────────────────────────────┘│ │ ││
│ │ │ │ │ ││
│ │ ▼ │ │ ││
│ │ ┌─────────────────────────────┐│ │ ││
│ │ │ build_fpn_mask_graph() ││ │ ││
│ │ │ ROI features (14x14) → ││ │ ││
│ │ │ 4x Conv3x3 → ││ │ ││
│ │ │ Deconv(28x28) → ││ │ ││
│ │ │ Conv1x1 + Sigmoid ││ │ ││
│ │ │ Output: mrcnn_mask [C] ││ │ ││
│ │ └─────────────────────────────┘│ │ ││
│ │ │ │ │ ││
│ │ ▼ │ │ ││
│ │ ┌─────────────────────────────┐│ │ ││
│ │ │ LOSS FUNCTIONS ││ │ ││
│ │ │ 1. rpn_class_loss ││ │ ││
│ │ │ 2. rpn_bbox_loss ││ │ ││
│ │ │ 3. mrcnn_class_loss ││ │ ││
│ │ │ 4. mrcnn_bbox_loss ││ │ ││
│ │ │ 5. mrcnn_mask_loss ││ │ ││
│ │ └─────────────────────────────┘│ │ ││
│ └─────────────────────────────────┘ └──────────────────────────────────────────────────────────┘│
│ │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘


---

## 🧩 Component Breakdown

### 1. ResNet Backbone (`resnet_graph`)

| Stage | Blocks | Output Shape | Purpose |
|-------|--------|--------------|---------|
| **Stage 1** | Conv7x7 + MaxPool | 256×256×64 | Initial feature extraction |
| **Stage 2** | Conv + 2x Identity | 256×256×256 | Fine details, edges |
| **Stage 3** | Conv + 3x Identity | 128×128×512 | Object parts, textures |
| **Stage 4** | Conv + 5/22x Identity | 64×64×1024 | Whole objects, structures |
| **Stage 5** | Conv + 2x Identity | 32×32×2048 | Semantic concepts, context |

**Key Components:**
- `identity_block`: Skip connection, no shape change
- `conv_block`: Skip connection with Conv1x1, changes shape
- **Both blocks have the SAME main path**, only shortcut differs

---

### 2. Feature Pyramid Network (FPN)

| Level | Resolution | Used For |
|-------|------------|----------|
| **P2** | 256×256 | Small objects (fine details) |
| **P3** | 128×128 | Medium objects |
| **P4** | 64×64 | Large objects |
| **P5** | 32×32 | Very large objects (semantic) |
| **P6** | 16×16 | RPN only (extra large) |

**Process:**
1. **Lateral Connections:** Conv1x1 on C2-C5 → 256 channels
2. **Top-Down Pathway:** Upsample + Add → Combine features
3. **Smoothing:** Conv3x3 on each P layer

---

### 3. Region Proposal Network (RPN)

| Component | Purpose | Output Shape |
|-----------|---------|--------------|
| **Shared Conv** | Feature extraction | `[batch, H, W, 512]` |
| **Classification Head** | FG vs BG scores | `[batch, total_anchors, 2]` |
| **BBox Head** | Anchor refinements | `[batch, total_anchors, 4]` |

- **Input:** P2-P6 feature maps
- **Output:** rpn_class_logits, rpn_probs, rpn_bbox
- **Anchors:** ~785,000 (5 scales × 3 ratios per pixel)

---

### 4. Proposal Layer

| Step | Operation | Output |
|------|-----------|--------|
| **1** | Select top 6000 by FG score | 6000 anchors |
| **2** | Apply deltas (`apply_box_deltas_graph`) | 6000 refined boxes |
| **3** | Clip to image (`clip_boxes_graph`) | 6000 clipped boxes |
| **4** | NMS (IoU > 0.7) | 2000/1000 proposals |

**Output:** `rpn_rois [batch, proposal_count, 4]`

---

### 5. DetectionTargetLayer (Training Only)

| Step | Operation | Purpose |
|------|-----------|---------|
| **1** | Remove padding | Clean data |
| **2** | Handle COCO crowds | Exclude crowds |
| **3** | Compute IoU | Match proposals to GT |
| **4** | Classify proposals | Positive (IoU≥0.5) / Negative (IoU<0.5) |
| **5** | Subsample to 200 | 33% positive, 67% negative |
| **6** | Generate targets | Class IDs, bbox deltas, masks |

**Output:** `rois [200,4]`, `target_class_ids [200]`, `target_bbox [200,4]`, `target_mask [200,28,28]`

---

### 6. PyramidROIAlign

| Step | Operation | Purpose |
|------|-----------|---------|
| **1** | Calculate area | Determine proposal size |
| **2** | Assign to FPN level | P2 (small), P3 (medium), P4 (large), P5 (very large) |
| **3** | Crop features | Extract from correct level |
| **4** | Resize to fixed size | 7×7 (classifier) or 14×14 (mask) |
| **5** | Bilinear interpolation | No quantization error |

**Output:** `[batch, num_rois, 7/14, 7/14, 256]`

---

### 7. Classifier + BBox Head (`fpn_classifier_graph`)

| Layer | Input | Output | Purpose |
|-------|-------|--------|---------|
| ROI Align (7×7) | `rois [200,4]` | `[200,7,7,256]` | Extract features |
| FC Layer 1 | `[200,7,7,256]` | `[200,1,1,1024]` | Feature transformation |
| FC Layer 2 | `[200,1,1,1024]` | `[200,1,1,1024]` | Feature transformation |
| Squeeze | `[200,1,1,1024]` | `[200,1024]` | Flatten |
| Classifier Head | `[200,1024]` | `[200,C]` | Predict class |
| BBox Head | `[200,1024]` | `[200,C,4]` | Predict deltas per class |

**Outputs:** `mrcnn_class_logits`, `mrcnn_probs`, `mrcnn_bbox`

---

### 8. Mask Head (`build_fpn_mask_graph`)

| Layer | Input | Output | Purpose |
|-------|-------|--------|---------|
| ROI Align (14×14) | `rois [200,4]` | `[200,14,14,256]` | Extract detailed features |
| Conv1-4 | `[200,14,14,256]` | `[200,14,14,256]` | Feature learning |
| Deconv | `[200,14,14,256]` | `[200,28,28,256]` | Upsample to 28×28 |
| Conv1×1 + Sigmoid | `[200,28,28,256]` | `[200,28,28,C]` | Predict per-class masks |

**Output:** `mrcnn_mask [batch, 200, 28, 28, C]`

---

### 9. DetectionLayer (Inference Only)

| Step | Operation | Purpose |
|------|-----------|---------|
| **1** | Get best class (argmax) | Choose class per proposal |
| **2** | Get class-specific deltas | Refine box for that class |
| **3** | Apply deltas | Refine boxes |
| **4** | Filter background | Remove class=0 |
| **5** | Filter low confidence | Remove low scores |
| **6** | Class-wise NMS | Remove duplicates per class |
| **7** | Pad to fixed size | Consistent output shape |

**Output:** `detections [batch, max_instances, 6]`

---

### 10. Loss Functions

| Loss | Type | Purpose |
|------|------|---------|
| **RPN Class** | Cross Entropy | Learn which anchors have objects |
| **RPN BBox** | Smooth L1 | Refine anchor boxes |
| **MRCNN Class** | Sparse Cross Entropy | Predict correct class |
| **MRCNN BBox** | Smooth L1 | Refine boxes per class |
| **MRCNN Mask** | Binary Cross Entropy | Predict accurate masks |

---

## 🔄 Training vs Inference Flow

### Training Mode
Image + GT → ResNet → FPN → RPN → Proposals (2000)
│
▼
DetectionTargetLayer → rois (200) + targets
│
▼
ROI Align → ROI features (7x7, 14x14)
│
▼
Heads → Predictions (class, bbox, mask)
│
▼
Loss Functions → Backpropagation → Update ALL weights

text

### Inference Mode
Image → ResNet → FPN → RPN → Proposals (1000)
│
▼
ROI Align → ROI features (7x7, 14x14)
│
▼
Heads → Predictions (class, bbox, mask)
│
▼
DetectionLayer → Final detections (boxes, classes, scores, masks)

text

---

## 📊 Data Flow Pipeline
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ COMPLETE DATA FLOW │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ │
│ DATASET │
│ ├── Images + Annotations (boxes, classes, masks) │
│ │ │
│ ▼ │
│ DataGenerator.getitem() │
│ ├── load_image_gt() → Image + GT (resized, augmented) │
│ ├── build_rpn_targets() → rpn_match + rpn_bbox │
│ └── Returns: [images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks] │
│ │ │
│ ▼ │
│ MODEL.TRAIN() │
│ ├── ResNet → C2, C3, C4, C5 │
│ ├── FPN → P2, P3, P4, P5, P6 │
│ ├── RPN → rpn_class, rpn_bbox │
│ ├── ProposalLayer → rpn_rois (2000/1000) │
│ ├── DetectionTargetLayer → rois (200) + targets │
│ ├── ROI Align → ROI features (7×7, 14×14) │
│ ├── Heads → Predictions │
│ ├── Loss → Compare predictions to targets │
│ └── Backpropagation → Update weights │
│ │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘


---

## ✨ Key Features

### Detection Capabilities
| Feature | Description |
|---------|-------------|
| **Bounding Box Detection** | Accurate localization with class labels |
| **Instance Segmentation** | Pixel-perfect masks for each object |
| **Multi-Object Detection** | Detect multiple objects of different classes |
| **Confidence Scoring** | Probability score for each detection |

### Technical Features
| Feature | Description |
|---------|-------------|
| **End-to-End Training** | Train all components together |
| **Multi-GPU Support** | Parallel training across GPUs |
| **Custom Dataset Support** | Easy integration with your own data |
| **Data Augmentation** | Built-in augmentation pipeline |
| **Transfer Learning** | Pre-trained ImageNet weights support |

---

## 🛠️ Technical Stack

### Core Dependencies
```txt
tensorflow-macos==2.14.0          # TensorFlow for M1 Mac
tensorflow-metal==1.0.1           # GPU acceleration for M1
keras==2.14.0                     # High-level API
numpy==1.23.5                     # Numerical operations
opencv-python==4.8.0.76           # Image operations
matplotlib==3.7.1                 # Visualization
scipy==1.11.3                     # Scientific computing
h5py==3.9.0                       # HDF5 file handling
imgaug==0.4.0                     # Data augmentation
Hardware Requirements
Component	Minimum	Recommended
GPU	8GB VRAM	12GB+ VRAM
RAM	16GB	32GB+
Storage	50GB	100GB+
CPU	4 cores	8+ cores

Installation

# 1. Create environment
conda create -n maskrcnn python=3.11
conda activate maskrcnn

# 2. Install TensorFlow (M1 Mac)
conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.14.0 tensorflow-metal==1.0.1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Clone repository
git clone https://github.com/yourusername/mask-rcnn-from-scratch.git
cd mask-rcnn-from-scratch
Inference
python
# 1. Create inference model
inference_model = MaskRCNN(mode='inference', config=config, model_dir='./logs')
inference_model.load_weights('path/to/weights.h5', by_name=True)

# 2. Detect
image = cv2.imread('test.jpg')
results = inference_model.detect([image])

# 3. Visualize
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], 
                           r['class_ids'], class_names, r['scores'])
📁 Project Structure
text
mask-rcnn-from-scratch/
│
├── README.md                       # This file
├── requirements.txt                 # Dependencies
│
├── mrcnn/                           # Core package
│   ├── model.py                     # Main implementation (3000+ lines)
│   ├── utils.py                     # Utility functions
│   ├── config.py                    # Configuration classes
│   └── parallel_model.py            # Multi-GPU support
│
├── examples/                        # Example notebooks
│   ├── train.ipynb                  # Training example
│   └── inference.ipynb              # Inference example
│
├── logs/                            # Training logs
│   └── mask_rcnn_YYYYMMDDTHHMM/     # Model checkpoints
│
└── weights/                         # Pre-trained weights
    └── mask_rcnn_coco.h5            # COCO pre-trained
🎓 Learning Outcomes
By studying this implementation, you will understand:

Concept	What You'll Learn
ResNet Architecture	How identity and conv blocks work together
Feature Pyramid Networks	Multi-scale feature extraction
Region Proposal Networks	Anchor-based object proposal generation
ROI Align	Precise feature extraction without quantization
Instance Segmentation	Pixel-perfect mask prediction
End-to-End Training	How all components train together
Loss Functions	Multi-task learning with 5 losses
Data Pipeline	Efficient data loading and augmentation
📄 License & Acknowledgments
License
MIT License - See LICENSE file for details.

Acknowledgments
Original Implementation: Matterport Mask R-CNN by Waleed Abdulla

Research Paper: Mask R-CNN by Kaiming He et al.

ResNet Code: Adapted from fchollet/deep-learning-models


🌟 If you find this project useful, please give it a star on GitHub!
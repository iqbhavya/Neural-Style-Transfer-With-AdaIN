# Neural Style Transfer with AdaIN

A PyTorch implementation of **Arbitrary Style Transfer using Adaptive Instance Normalization (AdaIN)**.

This project explores how a content image can be transformed using the visual characteristics of an arbitrary style image. The model uses a pretrained **VGG-19 encoder**, an **AdaIN transformation layer**, and a trainable **decoder** to generate stylized images.

## Overview

Traditional Neural Style Transfer methods often optimize an image iteratively for every new content-style pair. AdaIN takes a different approach.

The content and style images are passed through a pretrained VGG encoder. AdaIN aligns the channel-wise mean and variance of the content features with those of the style features. The transformed features are then passed through a decoder to reconstruct the stylized image.

```text
                    ┌─────────────────┐
Content Image ─────►│   VGG Encoder   │
                    └────────┬────────┘
                             │
                             │ Content Features
                             ▼
                        ┌─────────┐
Style Image ───────────►│  AdaIN  │
                        └────┬────┘
                             │
                             │ Stylized Features
                             ▼
                    ┌─────────────────┐
                    │     Decoder     │
                    └────────┬────────┘
                             │
                             ▼
                      Stylized Image
```

## What is AdaIN?

**Adaptive Instance Normalization (AdaIN)** transfers the statistical properties of a style image to the feature representation of a content image.

For a content feature map `x` and style feature map `y`, AdaIN can be expressed as:

```text
AdaIN(x, y) =
σ(y) * (x - μ(x)) / σ(x) + μ(y)
```

where:

- `μ(x)` = channel-wise mean of content features
- `σ(x)` = channel-wise standard deviation of content features
- `μ(y)` = channel-wise mean of style features
- `σ(y)` = channel-wise standard deviation of style features

This allows the model to transfer different styles without requiring a separate model for every style.

## Model Architecture

### 1. VGG-19 Encoder

A pretrained VGG-19 network is used as a fixed feature extractor.

The encoder extracts hierarchical representations from the input images:

```text
Image
  ↓
VGG Layers
  ↓
Deep Feature Representation
```

The encoder parameters are frozen during decoder training.

### 2. AdaIN

AdaIN receives content and style feature representations and aligns the content feature statistics with those of the style.

```text
Content Features
       │
       ├── Normalize using content statistics
       │
       └── Apply style statistics
                    │
                    ▼
             AdaIN Features
```

### 3. Decoder

The decoder learns to reconstruct an RGB image from the transformed feature representation.

It uses convolutional layers and upsampling operations to progressively recover the spatial resolution.

## Project Structure

```text
Neural-Style-Transfer-With-AdaIN/
│
├── content_data/
│   └── content images
│
├── style_data/
│   └── style images
│
├── experiment/
│   └── experiment configurations
│
├── utils/
│   ├── models.py
│   └── utils.py
│
├── train.py
│
├── vgg_normalised.pth
│
└── README.md
```

## Dataset

The project separates content and style images into two directories:

```text
content_data/
├── image1.jpg
├── image2.jpg
└── image3.jpg

style_data/
├── style1.jpg
├── style2.jpg
└── style3.jpg
```

The model can combine different content and style images without retraining the network for each pair.

## Requirements

Recommended environment:

- Python 3.10+
- PyTorch
- TorchVision
- Pillow
- NumPy
- CUDA-enabled PyTorch for GPU acceleration

Install the dependencies:

```bash
pip install torch torchvision pillow numpy
```

For GPU acceleration, make sure that the installed PyTorch build supports CUDA.

Check CUDA:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## GPU Support

The project automatically selects CUDA when an NVIDIA GPU is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Otherwise, it falls back to CPU.

Check the NVIDIA GPU from the terminal:

```bash
nvidia-smi
```

GPU acceleration is recommended because training the decoder with high-resolution feature maps can be computationally expensive.

## Training

Training is performed using content and style image datasets.

Example:

```bash
python train.py
```

Custom datasets can be provided using command-line arguments:

```bash
python train.py \
    --content_dir "./content_data" \
    --style_dir "./style_data" \
    --vgg "./vgg_normalised.pth" \
    --experiment "experiment1"
```

### Windows PowerShell

```powershell
python train.py --content_dir ".\content_data" --style_dir ".\style_data" --vgg ".\vgg_normalised.pth" --experiment "experiment1"
```

Additional parameters can be configured for:

- Content image size
- Style image size
- Final output size
- Batch size
- Learning rate
- Learning-rate decay
- Experiment name

## Experiments

Experiment-specific configurations are stored under:

```text
experiment/
```

For example:

```text
experiment/
└── experiment1/
    └── args.txt
```

This allows different training configurations to be tracked and compared.

## Inference

Once the decoder has been trained, a content image and an arbitrary style image can be passed through the model:

```text
Content Image + Style Image
          ↓
     VGG Encoder
          ↓
        AdaIN
          ↓
       Decoder
          ↓
    Stylized Image
```

The same trained decoder can be used with different content-style combinations.

## Key Concepts

This project demonstrates several deep learning concepts:

- Convolutional neural networks
- Transfer learning
- VGG feature extraction
- Feature-space image representation
- Instance normalization
- Adaptive Instance Normalization
- Encoder-decoder architectures
- Perceptual loss
- GPU-accelerated training
- Image-to-image transformation

## Why AdaIN?

A major advantage of AdaIN is that it supports **arbitrary style transfer**.

Instead of training a separate model for every artistic style, the same network can process different style images at inference time.

```text
One trained model
       │
       ├── Style A → Output A
       ├── Style B → Output B
       ├── Style C → Output C
       └── Style D → Output D
```

## Results

Example results will be added here after completing training and inference.

A useful comparison is:

| Content | Style | Output |
|--------|-------|--------|
| Original content | Artistic style | Stylized result |
| Original content | Different style | Stylized result |

## Current Status

- [x] Project structure
- [x] VGG-19 encoder
- [x] Content/style dataset loading
- [x] GPU device selection
- [x] Decoder architecture
- [ ] AdaIN layer
- [ ] Complete training loop
- [ ] Perceptual loss
- [ ] Decoder checkpoint saving
- [ ] Inference pipeline
- [ ] Generated image examples
- [ ] Quantitative/qualitative evaluation

## Future Improvements

- Add a complete inference script
- Add pretrained decoder checkpoints
- Add configurable style-strength control
- Add support for high-resolution inference
- Add automated result grids
- Add training-loss visualization
- Add a web interface for image upload
- Add comparisons with conventional Neural Style Transfer
- Benchmark inference speed on GPU

## Reference

This project is based on the AdaIN approach introduced in:

**Huang, X. and Belongie, S. — "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"**

The method introduced Adaptive Instance Normalization for real-time arbitrary style transfer.

## License

This project is intended for educational and research purposes.
Here’s a clean README you can use for the GitHub repo.

# Neural Style Transfer

A PyTorch implementation of **Neural Style Transfer (NST)** that combines the content of one image with the artistic style of another image using deep feature representations extracted from a pretrained VGG network.

## Overview

Neural Style Transfer is a deep learning technique that generates a new image by combining:

- **Content Image**: Provides the structure and objects of the final image.
- **Style Image**: Provides colors, textures, patterns, and artistic appearance.
- **Generated Image**: Combines the content and style representations.

The project uses a pretrained **VGG network** to extract high-level content features and low-level style features.

## How It Works

The model optimizes a generated image by minimizing two main losses:

### Content Loss

Content loss measures how different the generated image is from the content image in terms of deep feature representations.

### Style Loss

Style loss measures the difference between the style representations of the generated image and the style image. Style representations are calculated using **Gram matrices** of feature activations.

### Total Loss

The optimization objective can be represented as:

```text
Total Loss = α × Content Loss + β × Style Loss
```

where:

- `α` controls the importance of content
- `β` controls the importance of style

## Tech Stack

- Python
- PyTorch
- TorchVision
- VGG-19
- NumPy
- Pillow
- CUDA

## Project Structure

```text
NST/
│
├── content_data/
│   └── content images
│
├── style_data/
│   └── style images
│
├── experiment/
│   └── experiment1/
│       └── args.txt
│
├── vgg_normalised.pth
│
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

Python 3.10+ is recommended.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install a CUDA-enabled version of PyTorch compatible with your system.

Verify CUDA availability:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Dataset

Place your images in the corresponding directories:

```text
content_data/
style_data/
```

For example:

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

## Configuration

The project supports command-line arguments for specifying the content dataset, style dataset, pretrained VGG model, and experiment name.

Example:

```bash
python main.py \
    --content_dir "./content_data" \
    --style_dir "./style_data" \
    --vgg "./vgg_normalised.pth" \
    --experiment "experiment1"
```

On Windows PowerShell:

```powershell
python main.py --content_dir ".\content_data" --style_dir ".\style_data" --vgg ".\vgg_normalised.pth" --experiment "experiment1"
```

## GPU Support

The project automatically selects CUDA when an NVIDIA GPU is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Otherwise, it falls back to CPU.

Check your GPU:

```bash
nvidia-smi
```

## Experiments

Each experiment is stored separately:

```text
experiment/
├── experiment1/
├── experiment2/
└── experiment3/
```

This makes it easier to compare different:

- Content/style weights
- Optimization settings
- Learning rates
- Number of iterations
- Content and style images

## Results

Example output:

```text
Content Image + Style Image
           ↓
      VGG Feature
       Extraction
           ↓
   Content & Style Loss
           ↓
      Optimization
           ↓
    Generated Image
```

Results and experiment-specific configurations can be stored inside the corresponding experiment directory.

## Future Improvements

- [ ] Add complete NST optimization pipeline
- [ ] Add configurable content/style weights
- [ ] Add support for multiple style images
- [ ] Add image preprocessing and postprocessing
- [ ] Add training/inference progress visualization
- [ ] Add automatic result saving
- [ ] Add experiment metrics
- [ ] Add a web interface for uploading images
- [ ] Optimize inference using GPU acceleration

## References

The implementation is based on the ideas introduced in:

**Gatys et al., "A Neural Algorithm of Artistic Style"**

The project uses VGG feature representations to separate image content from artistic style.

## License

This project is intended for educational and research purposes.
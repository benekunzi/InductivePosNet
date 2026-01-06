# InductivePosNet: Deep Learning Positioning System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-Project-yellow)](https://wandb.ai/pos-ba)

InductivePosNet is a deep learning-based regression framework designed to predict spatial coordinates from inductive sensor measurement data. The project supports various neural network architectures, including Multi-Layer Perceptrons (MLP) and Residual Networks (ResNet), and is optimized for execution on high-performance computing clusters.

## What the project does

The core of the project is a regression model that takes measurement data from four sensors (A4T, A5T, A6T, and A7T) and predicts the corresponding $(x, y)$ coordinates. It handles the complete pipeline from raw CSV data processing and augmentation to model training and evaluation using real-world distance metrics.

### Key Features
- **Data Processing**: Automatic sorting and parsing of sensor measurement CSV files.
- **Data Augmentation**: Linear interpolation of measurement steps to increase dataset resolution.
- **Model Architectures**: Support for standard MLP and ResNet architectures with configurable depth and width.
- **Training Management**: Full integration with [Weights & Biases (WandB)](https://wandb.ai/) for experiment tracking, hyperparameter tuning, and model checkpointing.
- **Cluster Ready**: Prepared scripts for SLURM-based GPU clusters (e.g., NVIDIA DGX systems).
- **Custom Metrics**: Evaluation of model performance in real-world units (mm) using inverse-scaling metrics.

## Why the project is useful

- **Accuracy**: Leverages deep learning to map complex, non-linear sensor responses to precise coordinates.
- **Flexibility**: Highly configurable training parameters allows for rapid experimentation with different hyperparameters.
- **Scalability**: Designed to handle large-scale training jobs with batch scripts and automated logging.
- **Insight**: Includes custom metrics and Jupyter notebooks for detailed analysis of positioning errors across different ranges.

## How to get started

### Prerequisites

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy tensorflow scikit-learn wandb matplotlib
```

*Note: A `requirements.txt` file is recommended for production environments.*

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/benekunzi/InductivePosNet.git
   cd InductivePosNet
   ```

2. Set up your WandB account:
   ```bash
   wandb login
   ```

### Usage

#### Training a Model

You can train a baseline model using the default parameters:

```bash
python3 main.py --name "my-experiment" --distance 250 --epochs 100
```

For a more complex ResNet model on a cluster, you can use the provided bash scripts. For example:

```bash
sbatch bash_scripts/resnet.sh
```

#### Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--name` | `baseline` | WandB run name |
| `--distance` | `250` | Maximum coordinate range to include in training |
| `--n_neurons` | `300` | Number of neurons per hidden layer |
| `--n_layers` | `2` | Number of hidden layers |
| `--learning_rate` | `0.001` | Initial learning rate |
| `--epochs` | `180` | Number of training epochs |
| `--resnet` | `False` | Use ResNet architecture if `True` |
| `--regularization`| `False` | Apply L2 regularization |

#### Interactive Analysis

Use the `Positioning_Model.ipynb` notebook for data exploration and evaluating trained models visually.

## Project Structure

- `main.py`: Main entry point for training and data processing.
- `metrics.py`: Custom metrics for real-world coordinate evaluation.
- `bash_scripts/`: Collection of scripts for different training configurations and SLURM jobs.
- `01_measurement_data/`: Directory for input CSV measurement data.
- `trained_models/`: Storage for exported Keras models (`.h5`) and scalers (`.json`).
- `version_4.0/`: Advanced version containing updated logic and experimental features.

## Where users can get help

- **Issues**: Report bugs or request features via GitHub Issues.
- **Documentation**: Refer to the docstrings within `main.py` and `metrics.py` for API details.
- **Discussion**: For general questions, contact the maintainers.

## Who maintains and contributes

- **Maintainer**: [Benedict Kunzmann](https://github.com/benekunzi)
- **Contributors**: Contributions are welcome! Please see the guidelines below.

*For detailed contribution steps, please refer to `CONTRIBUTING.md` (if available).*

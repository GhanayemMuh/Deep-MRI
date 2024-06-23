# Deep Reconstruction of Knee MRI Images

![visualization_batch_0_rate_0 4](https://github.com/ahmadgh99/Deep-MRI/assets/61627067/5231a3c9-e2a1-449c-914d-f13fc4ce9342)

## Project Description

This project focuses on enhancing the quality and resolution of subsampled knee MRI images using a deep neural network based on a U-Net architecture. The primary goal is to reconstruct high-quality MRI images from subsampled data, aiding in improved diagnostic accuracy.

## Features

- **Deep Learning Model:** Utilizes a U-Net architecture tailored for medical image reconstruction.
- **Different Subsample Rates:** The ability to train the model on different subsample rates
- **MRI Image Enhancement:** Significantly improves the quality of subsampled knee MRI images.
- **Improved Diagnostics:** Provides clearer and more detailed images for better diagnosis.

## Installation

### Prerequisites

- Python 3.10
- Conda

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/ah/Deep-Reconstruction-Knee-MRI.git](https://github.com/ahmadgh99/Deep-MRI
    cd Deep-Reconstruction-Knee-MRI
    ```

2. Create and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate cs236781-miniproject
    ```

3. Run the Jupyter Notebook or Python scripts:
    - For Jupyter Notebook:
        ```bash
        jupyter notebook
        ```
    - For Python script:
        ```bash
        python main.py
        ```

### Example Command

```bash
python main.py --data-path './dataset/fastmri_knee' --batch-size 64 --num-workers 8 --num-epochs 25 --results-root './results' --device cuda:0 --lr 0.001
```

## Usage

1. **Data Preparation:** The dataset is not included in this repository. Please download the FASTMRI Knee dataset from Facebook
2. **Model Training:** Use the provided scripts or notebooks to train the model on your dataset.
3. **Reconstruction:** Run the trained model to reconstruct high-quality MRI images from subsampled inputs.

## Command-line Arguments

```python
parser.add_argument('--seed', type=int, default=0, help='Random Seed for reproducibility.')
parser.add_argument('--data-path', type=str, default='/datasets/fastmri_knee/', help='Path to MRI dataset.')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Use GPU if available')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--num-workers', type=int, default=1, help='Number of threads for data loading.')
parser.add_argument('--num-epochs', type=int, default=50, help='Total number of epochs for training.')
parser.add_argument('--report-interval', type=int, default=10, help='Interval for reporting training progress.')
parser.add_argument('--drop-rate', type=float, default=0.8, help='Drop rate for subsampling.')
parser.add_argument('--learn-mask', action='store_true', help='Flag to learn the subsampling mask.')
parser.add_argument('--results-root', type=str, default='results', help='Directory to save results.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--val-test-split', type=float, default=0.3, help='Split ratio for validation and test sets.')
parser.add_argument('--load-checkpoint', action='store_true', help='Load model from checkpoint if available.')
```

## Results

The reconstructed images show significant improvements in resolution and clarity, facilitating better diagnostic outcomes. Below are sample images comparing the original subsampled MRI and the reconstructed output.

### Visualization of Results

![visualization_batch_0_rate_0 6](https://github.com/ahmadgh99/Deep-MRI/assets/61627067/56ff6e4d-748d-42a0-a907-921af59a1970)

### PSNR over Epochs

![PSNR_train_val_rate_0 6](https://github.com/ahmadgh99/Deep-MRI/assets/61627067/d5bf76ce-2e95-4771-92f5-c0c4a0a89571)

### Loss over Epochs

![loss_over_epochs_rate_0 6](https://github.com/ahmadgh99/Deep-MRI/assets/61627067/c230f770-aad3-473c-abcf-8d5588155895)

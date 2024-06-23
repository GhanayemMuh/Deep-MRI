import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.mri_dataset import SliceData, DataTransform
from data import transforms
import matplotlib.pyplot as plt
import time
import numpy as np

def create_datasets(args, resolution=320):
    train_data = SliceData(
        root=f"{args.data_path}/singlecoil_train",
        transform=DataTransform(resolution),
        split=1
    )
    dev_data = SliceData(
        root=f"{args.data_path}/singlecoil_val",
        transform=DataTransform(resolution),
        split=args.val_test_split,
        validation=True
    )
    test_data = SliceData(
        root=f"{args.data_path}/singlecoil_val",
        transform=DataTransform(resolution),
        split=args.val_test_split,
        validation=False
    )
    return train_data, dev_data, test_data

def create_data_loaders(args):
    train_data, dev_data, test_data = create_datasets(args)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, dev_loader, test_loader

def freq_to_image(freq_data):
    ''' 
    This function accepts as input an image in the frequency domain, of size (B,320,320,2) (where B is batch size).
    Returns a tensor of size (B,320,320) representing the data in image domain.
    '''
    return transforms.complex_abs(transforms.ifft2_regular(freq_data))

def psnr(target, reconstruction):
    max_val = target.max()
    min_val = target.min()
    data_range = max_val - min_val

    data_range = data_range.item()

    mse = torch.mean((target - reconstruction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def evaluate_psnr(model, loader, device):
    psnr_values = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)

            for output, target in zip(outputs, targets):
                batch_psnr = psnr(target, output)
                if isinstance(batch_psnr, torch.Tensor):
                    batch_psnr = batch_psnr.item()
                psnr_values.append(batch_psnr)

    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    return mean_psnr, std_psnr

def visualize_sample(loader, model, device, results_root):
    print("Starting visualization...")
    data = next(iter(loader))
    inputs, targets = data
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()  
    with torch.no_grad():
        outputs, subsampled = model(inputs)

    outputs = outputs.cpu().detach().numpy()
    subsampled = subsampled.squeeze(1).cpu().detach().numpy()
    targets = targets.cpu().numpy()

    batch_index = 0
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(freq_to_image(inputs[batch_index]).cpu().numpy(), cmap='gray', aspect='auto')
    plt.title('Original')

    plt.subplot(1, 4, 2)
    plt.imshow(subsampled[batch_index], cmap='gray', aspect='auto')
    plt.title('Subsampled')

    plt.subplot(1, 4, 3)
    plt.imshow(outputs[batch_index], cmap='gray', aspect='auto')
    plt.title('Reconstruction')

    plt.subplot(1, 4, 4)
    plt.imshow(np.abs(np.squeeze(targets[batch_index]) - np.squeeze(outputs[batch_index])), cmap='gray', aspect='auto')
    plt.title('Difference')

    output_path = f"{results_root}/sample_visualization.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")


def visualize_multiple_samples(loader, model, device, results_root, num_samples=4, batch_index=0, drop_rate=None):
    print("Starting visualization of multiple samples...")

    for idx, data in enumerate(loader):
        if idx == batch_index:
            inputs, targets = data
            break
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()
    with torch.no_grad():
        outputs, subsampled = model(inputs)

    outputs = outputs.cpu().detach().numpy()
    subsampled = subsampled.squeeze(1).cpu().detach().numpy()
    targets = targets.cpu().numpy()

    num_samples = min(num_samples, len(inputs))

    plt.figure(figsize=(18, 6 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 4, 4 * i + 1)
        plt.imshow(freq_to_image(inputs[i]).cpu().numpy(), cmap='gray', aspect='auto')
        plt.title('Original')

        plt.subplot(num_samples, 4, 4 * i + 2)
        plt.imshow(subsampled[i], cmap='gray', aspect='auto')
        plt.title('Subsampled')

        plt.subplot(num_samples, 4, 4 * i + 3)
        plt.imshow(outputs[i], cmap='gray', aspect='auto')
        plt.title('Reconstruction')

        plt.subplot(num_samples, 4, 4 * i + 4)
        plt.imshow(np.abs(np.squeeze(targets[i]) - np.squeeze(outputs[i])), cmap='gray', aspect='auto')
        plt.title('Difference')

    output_path = f"{results_root}/visualization_batch_{batch_index}_rate_{drop_rate}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")


def log_training_details(epoch, num_epochs, iteration, total_iterations, loss, start_time, additional_info={}):
    elapsed_time = time.time() - start_time
    iteration_adjusted = (epoch * total_iterations) + iteration if epoch > 0 or iteration > 0 else 1
    estimated_total_time = elapsed_time / iteration_adjusted * (num_epochs * total_iterations)
    estimated_time_remaining = estimated_total_time - elapsed_time
    eta_hours, eta_remainder = divmod(estimated_time_remaining, 3600)
    eta_minutes, eta_seconds = divmod(eta_remainder, 60)
    log_message = (f"Epoch: [{epoch}/{num_epochs}], Step: [{iteration}/{total_iterations}], "
                   f"Loss: {loss:.4f}, ETA: {int(eta_hours):02}:{int(eta_minutes):02}:{int(eta_seconds):02}")
    for key, value in additional_info.items():
        log_message += f", {key}: {value:.4f}"
    print(log_message)

def log_validation_metrics(epoch, metrics):
    metrics_str = ', '.join(f"{key}: {value:.4f}" for key, value in metrics.items())
    print(f"Validation - After Epoch {epoch}: {metrics_str}")


def plot_psnr_over_epochs(train_psnr, val_psnr, rate, results_root):
    epochs = list(range(1, len(train_psnr) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_psnr, marker='o', linestyle='-', label='Train PSNR')
    plt.plot(epochs, val_psnr, marker='o', linestyle='-', label='Validation PSNR')
    plt.title(f'PSNR over Epochs for Subsampling Rate {rate}')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_root}/PSNR_train_val_rate_{rate}.png")
    plt.close()

def plot_train_val_comparison(train_psnr_values, val_psnr_values, rate, results_root):
    epochs = list(range(1, len(train_psnr_values) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_psnr_values, marker='o', linestyle='-', label='Train PSNR')
    plt.plot(epochs, val_psnr_values, marker='o', linestyle='-', label='Validation PSNR')
    plt.title(f'PSNR Comparison for Subsampling Rate {rate}')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_root}/PSNR_Comparison_rate_{rate}.png")
    plt.close()

def plot_loss_over_epochs(losses, rate, results_root):
  epochs = list(range(1, len(losses) + 1))
  plt.figure(figsize=(10, 5))
  plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
  plt.title(f'Loss over Epochs for Subsampling Rate {rate}')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.grid(True)
  plt.savefig(f"{results_root}/loss_over_epochs_rate_{rate}.png")
  plt.close()

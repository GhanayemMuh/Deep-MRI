import argparse
import random
import torch
import time
import numpy as np
from models.model import MyReconstructionModel
from utils.utils import *
import torch.nn.functional as F
import os


def main():
    args = create_arg_parser().parse_args()
    train_loader, validation_loader, test_loader = create_data_loaders(args)
    start_time = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    subsampling_rates = [0.2, 0.4, 0.6]
    results = []

    for rate in subsampling_rates:
        args.drop_rate = rate
        print(f"\nTraining, validating and testing at subsampling rate: {rate}")

        model = MyReconstructionModel(drop_rate=args.drop_rate, device=args.device, learn_mask=args.learn_mask).to(args.device)
        checkpoint_path = os.path.join(args.results_root, f'best_model_rate_{rate}.pth')

        if args.load_checkpoint and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded checkpoint for subsampling rate {rate}")
        else:
            print(f"No checkpoint found for subsampling rate {rate}, initializing new model.")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        best_val_psnr = float('-inf')
        patience = 10
        patience_counter = 0
        train_psnr_over_epochs = []
        val_psnr_over_epochs = []
        epoch_losses = []

        for epoch in range(args.num_epochs):
            model.train()
            total_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.learn_mask:
                    model.subsample.mask_grad(args.learn_mask)

                total_loss += loss.item()
                if i % args.report_interval == 0:
                    log_training_details(epoch, args.num_epochs, i, len(train_loader), loss.item(), start_time)

            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f'Average Loss after Epoch {epoch}: {avg_loss}')

            model.eval()
            avg_train_psnr, _ = evaluate_psnr(model, train_loader, args.device)
            avg_val_psnr, _ = evaluate_psnr(model, validation_loader, args.device)

            train_psnr_over_epochs.append(avg_train_psnr)
            val_psnr_over_epochs.append(avg_val_psnr)

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model checkpoint at subsampling rate {rate}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        plot_psnr_over_epochs(train_psnr_over_epochs, val_psnr_over_epochs, rate, args.results_root)
        plot_loss_over_epochs(epoch_losses, rate, args.results_root)

        best_epoch = max(zip(train_psnr_over_epochs, val_psnr_over_epochs), key=lambda x: x[1])
        print(f'Best Epoch PSNR: Train Mean: {best_epoch[0]}, Validation Mean: {best_epoch[1]}')

        visualize_multiple_samples(test_loader, model, args.device, args.results_root, num_samples=4, batch_index=0, drop_rate=rate)

        test_psnr, _ = evaluate_psnr(model, test_loader, args.device)
        results.append((rate, test_psnr))
        print(f"Test PSNR for subsampling rate {rate}: {test_psnr}")

    for rate, psnr in results:
        print(f"Final Test PSNR for subsampling rate {rate}: {psnr}")


def create_arg_parser():
    parser = argparse.ArgumentParser()
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
    return parser

if __name__ == "__main__":
    main()

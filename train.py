import argparse
import os
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DeblurDataset(Dataset):
    def __init__(self, blurred_dir, original_dir, files_blurred, files_original, resize=(256, 256)):
        self.blurred_dir = blurred_dir
        self.original_dir = original_dir
        self.files_blurred = [f for f in files_blurred if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.files_original = [f for f in files_original if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.length = min(len(self.files_blurred), len(self.files_original))
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        blurred_name = self.files_blurred[index]
        original_name = self.files_original[index]

        blurred = Image.open(os.path.join(self.blurred_dir, blurred_name)).convert("RGB")
        original = Image.open(os.path.join(self.original_dir, original_name)).convert("RGB")

        return self.transform(blurred), self.transform(original)

    def __len__(self):
        return self.length

def save_model(model, path):
    """Збереження ваг моделі."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model_name, device):
    """Завантаження pre-trained моделі (Restormer, MPRNet, FFTformer)."""
    from models.restormer import Restormer
    from models.mprnet import MPRNet
    from models.fftformer import fftformer

    if model_name == "restormer":
        model = Restormer()
    elif model_name == "mprnet":
        model = MPRNet()
    elif model_name == "fftformer":
        model = fftformer()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)

def train(model_name, blurred_dir, original_dir, epochs, batch_size, learning_rate,
           device, patience=3, min_delta=1e-4, val_split=0.2, random_state=42):

    files_blurred = sorted(os.listdir(blurred_dir))
    files_original = sorted(os.listdir(original_dir))

    count = min(len(files_blurred), len(files_original))
    files_blurred = files_blurred[:count]
    files_original = files_original[:count]

    train_blurred, val_blurred = train_test_split(
        files_blurred, test_size=val_split, random_state=random_state
    )
    train_original, val_original = train_test_split(
        files_original, test_size=val_split, random_state=random_state
    )

    resize_dim = (128, 128) if model_name == "fftformer" else (256, 256)

    train_dataset = DeblurDataset(blurred_dir, original_dir, train_blurred, train_original, resize=resize_dim)
    val_dataset = DeblurDataset(blurred_dir, original_dir, val_blurred, val_original, resize=resize_dim)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = load_model(model_name, device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = f"weights/{model_name}_finetuned_best.pth"

    train_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for blurred, original in tqdm(train_loader, desc=f"Train [{epoch}/{epochs}], Model [{model_name}]"):
            blurred, original = blurred.to(device), original.to(device)

            optimizer.zero_grad()
            output = model(blurred)

            if model_name == "mprnet":
                output = output[0]

            loss = criterion(output, original)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for blurred, original in val_loader:
                blurred, original = blurred.to(device), original.to(device)
                output = model(blurred)

                if model_name == "mprnet":
                    output = output[0]

                loss = criterion(output, original)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)

        print(f" Epoch [{epoch}/{epochs}], Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")

        if avg_val_loss + min_delta < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            save_model(model, best_model_path)
            print(f"New best model saved: {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title(f"Training and Validation Loss ({model_name})")
    plot_path = os.path.join(results_dir, f"{model_name}_loss.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

    results_file = os.path.join(results_dir, f"{model_name}_results.txt")
    with open(results_file, "w") as f:
        for i, (train_loss, val_loss, t) in enumerate(zip(train_losses, val_losses, epoch_times)):
            f.write(f"Epoch {i + 1}: Train_loss={train_loss:.6f}, Val_loss={val_loss:.6f}, Time={t:.2f}s\n")
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deblurring model with automatic train/val splitting.")
    parser.add_argument('--model', required=True, choices=['restormer', 'mprnet', 'fftformer'], help="Model to use.")
    parser.add_argument('--blurred_dir', required=True, help="Path to blurred images directory.")
    parser.add_argument('--original_dir', required=True, help="Path to original images directory.")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(
        model_name=args.model,
        blurred_dir=args.blurred_dir,
        original_dir=args.original_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        patience=args.patience,
        min_delta=args.min_delta,
        val_split=args.val_split
    )

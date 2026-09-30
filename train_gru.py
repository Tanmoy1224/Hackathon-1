import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# -----------------------------
# 1. Dataset Definition with Augmentation
# -----------------------------
class TemporalFatigueDataset(Dataset):
    def __init__(self, X, y, augment=False):
        # Shape: (N, 30, Features)
        self.X = torch.tensor(X, dtype=torch.float32)
        # Shape: (N,)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx].clone()

        # Apply synthetic eyeglass & lens glare augmentation during training
        if self.augment and np.random.rand() > 0.4:
            # 1. Compress EAR channels (indices 0, 1, 2) to simulate narrow glass frames
            scale = np.random.uniform(0.82, 0.94)
            sample[:, 0:3] *= scale

            # 2. Add subtle Gaussian noise to simulate glare/refraction variance
            glare_noise = torch.randn_like(sample[:, 0:3]) * 0.012
            sample[:, 0:3] += glare_noise

        return sample, self.y[idx]

# -----------------------------
# 2. Lightweight GRU Network
# -----------------------------
class LightweightGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=4, dropout=0.2):
        super(LightweightGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        # Extract features from the last temporal step (t=29)
        last_step_features = out[:, -1, :]
        logits = self.fc_block(last_step_features)
        return logits

# -----------------------------
# 3. Training Loop
# -----------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # Load pre-assembled temporal numpy matrices
    if not (os.path.exists("X_temporal.npy") and os.path.exists("y_temporal.npy")):
        raise FileNotFoundError("Run prepare_data.py first to generate X_temporal.npy and y_temporal.npy")

    X = np.load("X_temporal.npy")
    y = np.load("y_temporal.npy")

    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # Stratified Train/Val split (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Augmentation enabled on train set, disabled on validation set
    train_dataset = TemporalFatigueDataset(X_train, y_train, augment=True)
    val_dataset = TemporalFatigueDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(np.unique(y))
    model = LightweightGRU(input_size=X.shape[2], hidden_size=64, num_layers=2, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 25
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "fatigue_gru.pth")

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        print(f"Epoch [{epoch:02d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        # Save checkpoint with best validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint = {
                "state_dict": model.state_dict(),
                "input_size": X.shape[2],
                "hidden_size": 64,
                "num_layers": 2,
                "num_classes": num_classes
            }
            torch.save(checkpoint, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Acc: {best_val_acc:.2f}%)")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()
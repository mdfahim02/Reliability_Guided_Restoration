# ==========================================
# MODEL TRAINING
# ==========================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import RestorationDataset
from model import RestorationModel


# ------------------------------------------
# Dataset
# ------------------------------------------

dataset = RestorationDataset(
    "dataset/train/NoisyLR",
    "dataset/train/GT"
)

print("Total samples:", len(dataset))


# ------------------------------------------
# DataLoader
# ------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


# ------------------------------------------
# Model
# ------------------------------------------

model = RestorationModel()


# ------------------------------------------
# Loss function
# ------------------------------------------

criterion = nn.MSELoss()


# ------------------------------------------
# Optimizer
# ------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ------------------------------------------
# Training
# ------------------------------------------

epochs = 10

for epoch in range(epochs):

    total_loss = 0

    for noisy, gt in loader:

        # Model prediction
        output = model(noisy)

        # Calculate loss
        loss = criterion(output, gt)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        # Add loss
        total_loss += loss.item()


    # Average loss
    average_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {average_loss:.6f}"
    )


# ------------------------------------------
# Save trained model
# ------------------------------------------

torch.save(
    model.state_dict(),
    "restoration_model.pth"
)

print("\nTraining completed!")
print("Model saved as restoration_model.pth")
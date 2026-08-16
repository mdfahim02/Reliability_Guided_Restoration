# ==========================================
# RELIABILITY GUIDED RESTORATION MODEL
# ==========================================

# PyTorch import pannrom
import torch

# Neural network layers use panna
import torch.nn as nn


# ==========================================
# RESTORATION MODEL
# ==========================================

class RestorationModel(nn.Module):

    def __init__(self):

        # Parent class initialize pannrom
        super().__init__()

        # --------------------------------------
        # Feature extraction
        # --------------------------------------

        self.conv1 = nn.Conv2d(
            1,          # Input: grayscale image
            32,         # 32 feature maps
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        # --------------------------------------
        # Activation function
        # --------------------------------------

        self.relu = nn.ReLU()


        # --------------------------------------
        # 2x upscaling
        # --------------------------------------

        self.upsample = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=4,
            stride=2,
            padding=1
        )


        # --------------------------------------
        # Final restoration layer
        # --------------------------------------

        self.conv3 = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )


    # ==========================================
    # Forward pass
    # ==========================================

    def forward(self, x):

        # First feature extraction
        x = self.relu(self.conv1(x))

        # Second feature extraction
        x = self.relu(self.conv2(x))

        # 128x128 → 256x256
        x = self.relu(self.upsample(x))

        # Final restored image
        x = self.conv3(x)

        # Output range 0 to 1
        x = torch.sigmoid(x)

        return x
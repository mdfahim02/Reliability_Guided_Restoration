# ==========================================
# DATASET LOADER
# ==========================================

# os = folders and file paths handle panna
import os

# numpy = .npy files read panna
import numpy as np

# torch = AI model-ku tensors create panna
import torch

# Dataset = PyTorch dataset structure
from torch.utils.data import Dataset


# ==========================================
# RESTORATION DATASET CLASS
# ==========================================

class RestorationDataset(Dataset):

    # --------------------------------------
    # Constructor
    # --------------------------------------

    def __init__(self, noisy_folder, gt_folder):

        # NoisyLR folder path save pannrom
        self.noisy_folder = noisy_folder

        # GT folder path save pannrom
        self.gt_folder = gt_folder

        # NoisyLR folder-la irukkura .npy files
        self.files = sorted([
            f for f in os.listdir(noisy_folder)
            if f.endswith(".npy")
        ])


    # --------------------------------------
    # Total number of images
    # --------------------------------------

    def __len__(self):

        return len(self.files)


    # --------------------------------------
    # Get one image pair
    # --------------------------------------

    def __getitem__(self, index):

        # Filename edukkrom
        filename = self.files[index]

        # NoisyLR file path
        noisy_path = os.path.join(
            self.noisy_folder,
            filename
        )

        # GT file path
        gt_path = os.path.join(
            self.gt_folder,
            filename
        )

        # NoisyLR .npy load pannrom
        noisy = np.load(noisy_path)

        # GT .npy load pannrom
        gt = np.load(gt_path)

        # NumPy → PyTorch Tensor
        noisy = torch.from_numpy(noisy).float()

        # NumPy → PyTorch Tensor
        gt = torch.from_numpy(gt).float()

        # Channel dimension add pannrom
        # 128 × 128 → 1 × 128 × 128
        noisy = noisy.unsqueeze(0)

        # 256 × 256 → 1 × 256 × 256
        gt = gt.unsqueeze(0)

        # Input and target return pannrom
        return noisy, gt
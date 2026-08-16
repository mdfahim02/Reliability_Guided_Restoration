import os
import sys
import numpy as np
import torch

from model import RestorationModel


# ==========================================
# CHECK COMMAND-LINE ARGUMENTS
# ==========================================

if len(sys.argv) != 3:
    print("Usage:")
    print("python evaluation.py <test_images_directory> <output_directory>")
    sys.exit(1)


# ==========================================
# GET INPUT AND OUTPUT PATHS
# ==========================================

input_folder = sys.argv[1]
output_folder = sys.argv[2]


# ==========================================
# CHECK INPUT FOLDER
# ==========================================

if not os.path.isdir(input_folder):
    print("ERROR: Input folder not found:")
    print(input_folder)
    sys.exit(1)


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(output_folder, exist_ok=True)


# ==========================================
# SELECT DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# CREATE MODEL
# ==========================================

model = RestorationModel()

model.load_state_dict(
    torch.load(
        "restoration_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ==========================================
# FIND TEST IMAGES
# ==========================================

files = sorted([
    f for f in os.listdir(input_folder)
    if f.endswith(".npy")
])


if len(files) == 0:
    print("ERROR: No .npy files found in:")
    print(input_folder)
    sys.exit(1)


print("Total test images:", len(files))


# ==========================================
# RUN INFERENCE
# ==========================================

with torch.no_grad():

    for filename in files:

        # --------------------------------------
        # Load input image
        # --------------------------------------

        input_path = os.path.join(
            input_folder,
            filename
        )

        image = np.load(input_path)

        # --------------------------------------
        # Convert NumPy → Tensor
        # --------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        # --------------------------------------
        # Add batch and channel dimensions
        #
        # H × W
        # ↓
        # 1 × 1 × H × W
        # --------------------------------------

        image = image.unsqueeze(0).unsqueeze(0)

        image = image.to(device)

        # --------------------------------------
        # Model inference
        # --------------------------------------

        restored = model(image)

        # --------------------------------------
        # Tensor → NumPy
        # --------------------------------------

        restored = restored.squeeze().cpu().numpy()

        # --------------------------------------
        # Keep values between 0 and 1
        # --------------------------------------

        restored = np.clip(
            restored,
            0,
            1
        )

        # --------------------------------------
        # Save restored image
        # --------------------------------------

        output_path = os.path.join(
            output_folder,
            filename
        )

        np.save(
            output_path,
            restored
        )


print("================================")
print("Evaluation completed!")
print("Input folder :", input_folder)
print("Output folder:", output_folder)
print("Images processed:", len(files))
print("================================")
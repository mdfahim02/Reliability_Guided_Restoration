import os
import numpy as np

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ==========================================
# FOLDERS
# ==========================================

gt_folder = "dataset/train/GT"
restored_folder = "outputs"


# ==========================================
# FIND RESTORED FILES
# ==========================================

files = sorted([
    f for f in os.listdir(restored_folder)
    if f.endswith(".npy")
])


print("Total images:", len(files))


# ==========================================
# CALCULATE PSNR + SSIM
# ==========================================

total_psnr = 0.0
total_ssim = 0.0


for filename in files:

    # Ground Truth path
    gt_path = os.path.join(
        gt_folder,
        filename
    )

    # Restored path
    restored_path = os.path.join(
        restored_folder,
        filename
    )


    # Load images
    gt = np.load(gt_path)
    restored = np.load(restored_path)


    # Make sure values are between 0 and 1
    gt = np.clip(gt, 0, 1)
    restored = np.clip(restored, 0, 1)


    # --------------------------------------
    # PSNR
    # --------------------------------------

    psnr = peak_signal_noise_ratio(
        gt,
        restored,
        data_range=1.0
    )


    # --------------------------------------
    # SSIM
    # --------------------------------------

    ssim = structural_similarity(
        gt,
        restored,
        data_range=1.0
    )


    total_psnr += psnr
    total_ssim += ssim


    print(
        f"{filename}  "
        f"PSNR: {psnr:.2f} dB  "
        f"SSIM: {ssim:.4f}"
    )


# ==========================================
# AVERAGE RESULTS
# ==========================================

average_psnr = total_psnr / len(files)
average_ssim = total_ssim / len(files)


print("\n================================")
print("FINAL RESULTS")
print("================================")

print(f"Average PSNR : {average_psnr:.2f} dB")
print(f"Average SSIM : {average_ssim:.4f}")
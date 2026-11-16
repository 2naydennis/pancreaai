# make_slices.py: cuts the scans into ready slices and saves them.
# Run: python scripts/make_slices.py   (takes a few minutes and stays quiet)
# Saves: data/slices/train.npz and data/slices/val.npz

import os
import json
import random
import numpy as np
import SimpleITK as sitk

folder = "data/Task07_Pancreas"
out_folder = "data/slices"
rng = random.Random(0)    # Makes the same random picks every time.

# A recipe that gets one slice ready for the model.
def prepare(scan_slice):
    windowed = scan_slice.clip(-100, 240)       # Keep only the soft-organ numbers.
    normalised = (windowed + 100) / 340         # Make the numbers go from 0 to 1.
    small = normalised[::2, ::2]                # Make the picture half as wide.
    return (small * 255).astype(np.uint8)       # Store it small to save space.

with open("splits.json") as f:
    splits = json.load(f)

os.makedirs(out_folder, exist_ok=True)

# Only train and val. The test group stays locked until the end.
for group in ["train", "val"]:
    images, masks, patients = [], [], []
    for name in splits[group]:
        scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
        mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

        # Find which slices show the pancreas and which do not.
        has_pancreas = (mask_grid > 0).any(axis=(1, 2))
        with_pancreas = [i for i in range(len(has_pancreas)) if has_pancreas[i]]
        without = [i for i in range(len(has_pancreas)) if not has_pancreas[i]]

        # Keep all pancreas slices, plus the same number of empty ones.
        extra = rng.sample(without, min(len(with_pancreas), len(without)))

        for i in sorted(with_pancreas + extra):
            images.append(prepare(scan_grid[i]))
            masks.append((mask_grid[i][::2, ::2] > 0).astype(np.uint8))    # 1 means pancreas.
            patients.append(name)

    # Stack all the slices into one pile and save it.
    images = np.stack(images)
    masks = np.stack(masks)
    np.savez_compressed(out_folder + "/" + group + ".npz", images=images, masks=masks, patients=np.array(patients))
    print(group, "slices:", len(images), " with pancreas:", int(masks.any(axis=(1, 2)).sum()), " picture size:", images.shape[1:])

print("Saved to", out_folder)

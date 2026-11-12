import os
import json
import random
import numpy as np
import SimpleITK as sitk

folder = "data/Task07_Pancreas"
out_folder = "data/slices"
rng = random.Random(0)

# The same three moves as preprocess_demo.py: window, normalise, shrink
def prepare(scan_slice):
    windowed = scan_slice.clip(-100, 240)
    normalised = (windowed + 100) / 340
    small = normalised[::2, ::2]
    return (small * 255).astype(np.uint8)

with open("splits.json") as f:
    splits = json.load(f)

os.makedirs(out_folder, exist_ok=True)

# Only train and val. The test group stays locked until the very end.
for group in ["train", "val"]:
    images, masks, patients = [], [], []
    for name in splits[group]:
        scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
        mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

        # Which slices show pancreas, and which do not?
        has_pancreas = (mask_grid > 0).any(axis=(1, 2))
        with_pancreas = [i for i in range(len(has_pancreas)) if has_pancreas[i]]
        without = [i for i in range(len(has_pancreas)) if not has_pancreas[i]]

        # Keep every pancreas slice, plus the same number of empty slices
        extra = rng.sample(without, min(len(with_pancreas), len(without)))

        for i in sorted(with_pancreas + extra):
            images.append(prepare(scan_grid[i]))
            masks.append((mask_grid[i][::2, ::2] > 0).astype(np.uint8))
            patients.append(name)

    images = np.stack(images)
    masks = np.stack(masks)
    np.savez_compressed(out_folder + "/" + group + ".npz", images=images, masks=masks, patients=np.array(patients))
    print(group, "slices:", len(images), " with pancreas:", int(masks.any(axis=(1, 2)).sum()), " picture size:", images.shape[1:])

print("Saved to", out_folder)

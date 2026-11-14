# make_slices_3.py
# Same as make_slices.py with one change: each example holds three slices
# (the one below, the slice itself, the one above) instead of one.
# Run: python scripts/make_slices_3.py   (about 1.5 minutes, silent until a group is done)
# Saves: data/slices/train3.npz and data/slices/val3.npz
# Experiment 5 tests whether seeing the neighbour slices helps the model follow the pancreas.

import os
import json
import random
import numpy as np
import SimpleITK as sitk

folder = "data/Task07_Pancreas"
out_folder = "data/slices"

# The same seed as make_slices.py, so exactly the same slices get picked.
# That keeps the comparison fair: only the two extra neighbour slices are new.
rng = random.Random(0)     # same seed as make_slices.py, so the same slices get picked

# The same three moves as before: window, normalise, shrink.
def prepare(scan_slice):
    windowed = scan_slice.clip(-100, 240)
    normalised = (windowed + 100) / 340
    small = normalised[::2, ::2]
    return (small * 255).astype(np.uint8)

with open("splits.json") as f:
    splits = json.load(f)

os.makedirs(out_folder, exist_ok=True)

for group in ["train", "val"]:
    images, masks, patients = [], [], []
    for name in splits[group]:
        scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
        mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

        # The number of the last slice. Python counts from 0, so 90 slices are numbered 0 to 89.
        last = len(scan_grid) - 1

        # Pick the slices exactly as make_slices.py does: all pancreas slices plus as many empty ones.
        has_pancreas = (mask_grid > 0).any(axis=(1, 2))
        with_pancreas = [i for i in range(len(has_pancreas)) if has_pancreas[i]]
        without = [i for i in range(len(has_pancreas)) if not has_pancreas[i]]
        extra = rng.sample(without, min(len(with_pancreas), len(without)))

        for i in sorted(with_pancreas + extra):
            # THE ONE CHANGE: store three slices instead of one.
            # The first and last slice have no neighbour on one side.
            # max and min stop the number from leaving the scan, so the slice is used twice there.
            below = prepare(scan_grid[max(i - 1, 0)])
            middle = prepare(scan_grid[i])
            above = prepare(scan_grid[min(i + 1, last)])

            # Pile the three pictures together. One example now has the shape (3, 256, 256).
            images.append(np.stack([below, middle, above]))

            # The answer is still only the mask of the MIDDLE slice.
            masks.append((mask_grid[i][::2, ::2] > 0).astype(np.uint8))
            patients.append(name)

    # Save under a new name (train3.npz, val3.npz). The old files are not touched.
    images = np.stack(images)
    masks = np.stack(masks)
    np.savez_compressed(out_folder + "/" + group + "3.npz", images=images, masks=masks, patients=np.array(patients))
    print(group, "examples:", len(images), " with pancreas:", int(masks.any(axis=(1, 2)).sum()), " shape of one example:", images.shape[1:], flush=True)

print("Saved to", out_folder)

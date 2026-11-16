# make_slices_3.py: like make_slices.py, but each example has 3 slices.
# The model gets the slice below, the slice itself, and the slice above.
# Run: python scripts/make_slices_3.py   (takes about 1.5 minutes and stays quiet)
# Saves: data/slices/train3.npz and data/slices/val3.npz

import os
import json
import random
import numpy as np
import SimpleITK as sitk

folder = "data/Task07_Pancreas"
out_folder = "data/slices"
rng = random.Random(0)    # Same picks as make_slices.py, so the test is fair.

# A recipe that gets one slice ready for the model.
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
        last = len(scan_grid) - 1    # The number of the last slice.

        # Pick the same slices as before.
        has_pancreas = (mask_grid > 0).any(axis=(1, 2))
        with_pancreas = [i for i in range(len(has_pancreas)) if has_pancreas[i]]
        without = [i for i in range(len(has_pancreas)) if not has_pancreas[i]]
        extra = rng.sample(without, min(len(with_pancreas), len(without)))

        for i in sorted(with_pancreas + extra):
            # The new part: take 3 slices. At the edge, the slice is used twice.
            below = prepare(scan_grid[max(i - 1, 0)])
            middle = prepare(scan_grid[i])
            above = prepare(scan_grid[min(i + 1, last)])
            images.append(np.stack([below, middle, above]))

            # The answer is only the mask of the middle slice.
            masks.append((mask_grid[i][::2, ::2] > 0).astype(np.uint8))
            patients.append(name)

    images = np.stack(images)
    masks = np.stack(masks)
    np.savez_compressed(out_folder + "/" + group + "3.npz", images=images, masks=masks, patients=np.array(patients))
    print(group, "examples:", len(images), " with pancreas:", int(masks.any(axis=(1, 2)).sum()), " shape of one example:", images.shape[1:], flush=True)

print("Saved to", out_folder)

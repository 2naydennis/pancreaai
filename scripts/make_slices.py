# make_slices.py
# Cuts the training and validation scans into prepared 2D slices and saves them.
# Run: python scripts/make_slices.py   (a few minutes, silent until a group is done)
# Saves: data/slices/train.npz and data/slices/val.npz
# Preparing the slices once means training does not have to reopen 239 big scan files every epoch.

import os
import json
import random
import numpy as np
import SimpleITK as sitk

folder = "data/Task07_Pancreas"
out_folder = "data/slices"

# A random picker with a fixed seed, so it makes the same "random" picks every time.
rng = random.Random(0)

# A function is a reusable recipe: a slice goes in, the prepared slice comes out.
# It does the same three moves as preprocess_demo.py: window, normalise, shrink.
def prepare(scan_slice):
    windowed = scan_slice.clip(-100, 240)
    normalised = (windowed + 100) / 340
    small = normalised[::2, ::2]
    # Store as whole numbers 0..255 instead of 0..1. That keeps the saved file small.
    return (small * 255).astype(np.uint8)

with open("splits.json") as f:
    splits = json.load(f)

os.makedirs(out_folder, exist_ok=True)

# Only train and val. The test group stays locked until the very end.
for group in ["train", "val"]:
    # Three lists that fill up together: the slice, its mask, and the patient it came from.
    images, masks, patients = [], [], []
    for name in splits[group]:
        scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
        mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

        # Which slices show pancreas, and which do not?
        # has_pancreas holds one True or False per slice. The two lists hold the slice numbers.
        has_pancreas = (mask_grid > 0).any(axis=(1, 2))
        with_pancreas = [i for i in range(len(has_pancreas)) if has_pancreas[i]]
        without = [i for i in range(len(has_pancreas)) if not has_pancreas[i]]

        # Keep every pancreas slice, plus the same number of empty slices picked at random.
        # With all the empty slices included, the model would mostly learn to draw nothing.
        extra = rng.sample(without, min(len(with_pancreas), len(without)))

        for i in sorted(with_pancreas + extra):
            images.append(prepare(scan_grid[i]))

            # The mask is shrunk the same way. "> 0" merges pancreas (1) and tumor (2) into one label.
            masks.append((mask_grid[i][::2, ::2] > 0).astype(np.uint8))
            patients.append(name)

    # Pile the separate slices into one big block and save it as one compressed file.
    images = np.stack(images)
    masks = np.stack(masks)
    np.savez_compressed(out_folder + "/" + group + ".npz", images=images, masks=masks, patients=np.array(patients))
    print(group, "slices:", len(images), " with pancreas:", int(masks.any(axis=(1, 2)).sum()), " picture size:", images.shape[1:])

print("Saved to", out_folder)

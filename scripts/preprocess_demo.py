# preprocess_demo.py: shows how a slice gets ready before the model sees it.
# Run: python scripts/preprocess_demo.py
# Saves: outputs/preprocess_demo.png

import SimpleITK as sitk
import matplotlib.pyplot as plt

folder = "data/Task07_Pancreas"
name = "pancreas_001.nii.gz"

# Load the scan and the mask.
scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

# Pick the slice with the most pancreas and turn it upside down.
best = (mask_grid > 0).sum(axis=(1, 2)).argmax()
raw = scan_grid[best][::-1, ::-1]
mask = mask_grid[best][::-1, ::-1]

# Step 1: keep only the numbers from -100 to 240, where soft organs are.
windowed = raw.clip(-100, 240)

# Step 2: change the numbers so they go from 0 to 1.
normalised = (windowed + 100) / 340

# Step 3: keep every second pixel, so the picture becomes half as wide.
small = normalised[::2, ::2]
small_mask = mask[::2, ::2] > 0    # True means pancreas.

print("Raw:        size", raw.shape, " lowest", raw.min(), " highest", raw.max())
print("Windowed:   size", windowed.shape, " lowest", windowed.min(), " highest", windowed.max())
print("Normalised: size", normalised.shape, " lowest", normalised.min(), " highest", normalised.max())
print("Small:      size", small.shape, " lowest", small.min(), " highest", small.max())
print("Pancreas pixels:", small_mask.sum(), "out of", small_mask.size)
print("Pancreas share of the picture (%):", round(100 * small_mask.sum() / small_mask.size, 2))    # The pancreas is tiny.

# Draw the three stages next to each other.
fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 5))
a.imshow(raw, cmap="gray")
a.set_title("Raw scan: every number shown")
b.imshow(small, cmap="gray")
b.set_title("What the model sees")
c.imshow(small_mask, cmap="gray")
c.set_title("What the model must draw")
for panel in (a, b, c):
    panel.axis("off")
plt.savefig("outputs/preprocess_demo.png", dpi=100, bbox_inches="tight")
print("Saved picture")

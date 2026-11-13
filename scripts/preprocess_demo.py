# preprocess_demo.py
# Shows the three steps every slice goes through before the model sees it:
# window, normalise, shrink. Preparing data like this is called preprocessing.
# Run: python scripts/preprocess_demo.py
# Saves: outputs/preprocess_demo.png

import SimpleITK as sitk
import matplotlib.pyplot as plt

folder = "data/Task07_Pancreas"
name = "pancreas_001.nii.gz"

# 1. Load one scan and its mask, as grids of numbers with directions [slice, row, column].
scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

# 2. Pick the slice showing the most pancreas (tumor counts as part of it).
# Count the marked voxels in each slice, then take the position of the biggest count.
best = (mask_grid > 0).sum(axis=(1, 2)).argmax()

# Take that slice from the scan and from the mask. [::-1, ::-1] turns it 180 degrees.
raw = scan_grid[best][::-1, ::-1]
mask = mask_grid[best][::-1, ::-1]

# 3. Window: any number below -100 becomes -100, any number above 240 becomes 240.
# Air and bone lose their detail, which we do not need. Soft organs keep all of theirs.
windowed = raw.clip(-100, 240)

# 4. Normalise: stretch -100..240 so it becomes 0..1.
# Neural networks learn best from small numbers like these.
normalised = (windowed + 100) / 340

# 5. Shrink: keep every second pixel, so 512x512 becomes 256x256.
# A quarter of the pixels means much faster training.
small = normalised[::2, ::2]

# Shrink the mask the same way. "> 0" turns it into True (pancreas or tumor) and False.
small_mask = mask[::2, ::2] > 0

# Print the size and the number range after each step, to see what changed.
print("Raw:        size", raw.shape, " lowest", raw.min(), " highest", raw.max())
print("Windowed:   size", windowed.shape, " lowest", windowed.min(), " highest", windowed.max())
print("Normalised: size", normalised.shape, " lowest", normalised.min(), " highest", normalised.max())
print("Small:      size", small.shape, " lowest", small.min(), " highest", small.max())

# The pancreas fills only about 1.5% of the picture.
# That is why accuracy is a useless score here and we use Dice instead.
print("Pancreas pixels:", small_mask.sum(), "out of", small_mask.size)
print("Pancreas share of the picture (%):", round(100 * small_mask.sum() / small_mask.size, 2))

# 6. Draw the three stages side by side. a, b and c are the three picture areas.
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

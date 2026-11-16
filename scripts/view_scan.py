# view_scan.py: opens one CT scan, measures the tumor, and saves a picture.
# Run: python scripts/view_scan.py
# In the mask, 0 means nothing, 1 means pancreas, and 2 means tumor.

import SimpleITK as sitk           # Reads CT scan files.
import matplotlib.pyplot as plt    # Draws pictures.

folder = "data/Task07_Pancreas"
name = "pancreas_001.nii.gz"

# Load the scan and the doctor's mask.
scan = sitk.ReadImage(folder + "/imagesTr/" + name)
mask = sitk.ReadImage(folder + "/labelsTr/" + name)

print("Size (voxels):", scan.GetSize())       # How many tiny bricks the scan has.
print("Spacing (mm): ", scan.GetSpacing())    # How big one brick is.

# Turn both into grids of numbers we can work with.
scan_grid = sitk.GetArrayFromImage(scan)
mask_grid = sitk.GetArrayFromImage(mask)

print("Lowest number: ", scan_grid.min())
print("Highest number:", scan_grid.max())

# Count the tumor bricks, then multiply by the size of one brick.
tumor_voxels = (mask_grid == 2).sum()
sx, sy, sz = scan.GetSpacing()
one_voxel = sx * sy * sz
print("Tumor voxels:", tumor_voxels)
print("One voxel (mm3):", round(one_voxel, 3))
print("Tumor volume (mm3):", round(tumor_voxels * one_voxel))

# Find the slice that shows the most tumor.
best = (mask_grid == 2).sum(axis=(1, 2)).argmax()
print("Slice with most tumor:", best)

# Turn the slice upside down, because that is how doctors look at CT scans.
scan_slice = scan_grid[best][::-1, ::-1]
mask_slice = mask_grid[best][::-1, ::-1]

# Draw two pictures side by side: the plain slice, and the slice with outlines.
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6))
left.imshow(scan_slice, cmap="gray", vmin=-160, vmax=240)
left.set_title("CT slice " + str(best))
right.imshow(scan_slice, cmap="gray", vmin=-160, vmax=240)
right.contour(mask_slice == 1, colors="lime", linewidths=1)    # Green line around the pancreas.
right.contour(mask_slice == 2, colors="red", linewidths=1)     # Red line around the tumor.
right.set_title("Green = pancreas, red = tumor")
left.axis("off")
right.axis("off")
plt.savefig("outputs/first_scan.png", dpi=100, bbox_inches="tight")
print("Saved picture")

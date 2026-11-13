# view_scan.py
# Opens one CT scan and the doctor's mask, measures the tumor, saves a picture.
# Run: python scripts/view_scan.py
# Saves: outputs/first_scan.png
# The mask has the same size as the scan. Each voxel holds 0 (background),
# 1 (pancreas) or 2 (tumor).

# SimpleITK reads medical image files. matplotlib draws pictures.
# "as sitk" and "as plt" are short nicknames for the two libraries.
import SimpleITK as sitk
import matplotlib.pyplot as plt

folder = "data/Task07_Pancreas"
name = "pancreas_001.nii.gz"

# 1. Load the scan and the doctor's mask.
# The + sign glues pieces of text together into the full path of the file.
scan = sitk.ReadImage(folder + "/imagesTr/" + name)
mask = sitk.ReadImage(folder + "/labelsTr/" + name)

# 2. Print the facts stored in the file.
# Size is the number of voxels in each direction. Spacing is the size of one voxel in mm.
print("Size (voxels):", scan.GetSize())
print("Spacing (mm): ", scan.GetSpacing())

# 3. Turn both into plain grids of numbers that we can do maths on.
# The grid has three directions: [slice, row, column].
scan_grid = sitk.GetArrayFromImage(scan)
mask_grid = sitk.GetArrayFromImage(mask)

print("Lowest number: ", scan_grid.min())
print("Highest number:", scan_grid.max())

# 4. Count the tumor voxels and turn the count into a volume.
# (mask_grid == 2) is True wherever the mask says tumor, and .sum() counts the Trues.
tumor_voxels = (mask_grid == 2).sum()

# The volume of one voxel is width * height * depth.
sx, sy, sz = scan.GetSpacing()
one_voxel = sx * sy * sz
print("Tumor voxels:", tumor_voxels)
print("One voxel (mm3):", round(one_voxel, 3))
print("Tumor volume (mm3):", round(tumor_voxels * one_voxel))

# 5. Find the slice with the most tumor in it.
# .sum(axis=(1, 2)) counts the tumor voxels inside each slice.
# .argmax() gives the position of the biggest count.
best = (mask_grid == 2).sum(axis=(1, 2)).argmax()
print("Slice with most tumor:", best)

# 6. Draw that slice.
# [::-1, ::-1] flips it both ways, which turns it 180 degrees, the way doctors view a CT.
scan_slice = scan_grid[best][::-1, ::-1]
mask_slice = mask_grid[best][::-1, ::-1]

# Two pictures side by side: the plain slice on the left, the same slice with outlines on the right.
# vmin and vmax set which numbers are drawn black and white, so soft organs show clearly.
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6))
left.imshow(scan_slice, cmap="gray", vmin=-160, vmax=240)
left.set_title("CT slice " + str(best))
right.imshow(scan_slice, cmap="gray", vmin=-160, vmax=240)

# contour draws a line around a region: green around the pancreas (1), red around the tumor (2).
right.contour(mask_slice == 1, colors="lime", linewidths=1)
right.contour(mask_slice == 2, colors="red", linewidths=1)
right.set_title("Green = pancreas, red = tumor")
left.axis("off")
right.axis("off")
plt.savefig("outputs/first_scan.png", dpi=100, bbox_inches="tight")
print("Saved picture")

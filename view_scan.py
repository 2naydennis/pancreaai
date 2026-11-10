import SimpleITK as sitk
import matplotlib.pyplot as plt

folder = "data/Task07_Pancreas"
name = "pancreas_001.nii.gz"

# 1. Load the scan and the doctor's mask
scan = sitk.ReadImage(folder + "/imagesTr/" + name)
mask = sitk.ReadImage(folder + "/labelsTr/" + name)

# 2. Print the facts stored in the file
print("Size (voxels):", scan.GetSize())
print("Spacing (mm): ", scan.GetSpacing())

# 3. Turn both into plain grids of numbers
scan_grid = sitk.GetArrayFromImage(scan)
mask_grid = sitk.GetArrayFromImage(mask)

print("Lowest number: ", scan_grid.min())
print("Highest number:", scan_grid.max())

# 4. Count bricks and measure the tumor
tumor_voxels = (mask_grid == 2).sum()
sx, sy, sz = scan.GetSpacing()
one_voxel = sx * sy * sz
print("Tumor voxels:", tumor_voxels)
print("One voxel (mm3):", round(one_voxel, 3))
print("Tumor volume (mm3):", round(tumor_voxels * one_voxel))

# 5. Find the slice with the most tumor in it
best = (mask_grid == 2).sum(axis=(1, 2)).argmax()
print("Slice with most tumor:", best)

# 6. Draw that slice, turned the way doctors view it
scan_slice = scan_grid[best][::-1, ::-1]
mask_slice = mask_grid[best][::-1, ::-1]

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6))
left.imshow(scan_slice, cmap="gray", vmin=-160, vmax=240)
left.set_title("CT slice " + str(best))
right.imshow(scan_slice, cmap="gray", vmin=-160, vmax=240)
right.contour(mask_slice == 1, colors="lime", linewidths=1)
right.contour(mask_slice == 2, colors="red", linewidths=1)
right.set_title("Green = pancreas, red = tumor")
left.axis("off")
right.axis("off")
plt.savefig("first_scan.png", dpi=100, bbox_inches="tight")
print("Saved picture")

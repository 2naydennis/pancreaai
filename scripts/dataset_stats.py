# dataset_stats.py: measures the pancreas and tumor size of all 281 patients.
# Run: python scripts/dataset_stats.py
# Saves: outputs/dataset_stats.csv

import csv
import json
import SimpleITK as sitk

folder = "data/Task07_Pancreas/labelsTr"

# Load the three patient groups.
with open("splits.json") as f:
    splits = json.load(f)

# Go through every patient and measure them.
rows = []
for group in ["train", "val", "test"]:
    for name in splits[group]:
        mask = sitk.ReadImage(folder + "/" + name)
        sx, sy, sz = mask.GetSpacing()
        one_voxel = sx * sy * sz                       # The size of one tiny brick.
        grid = sitk.GetArrayFromImage(mask)
        pancreas_mm3 = (grid == 1).sum() * one_voxel   # Pancreas bricks times brick size.
        tumor_mm3 = (grid == 2).sum() * one_voxel      # Tumor bricks times brick size.
        rows.append([name, group, mask.GetSize()[2], round(sz, 2), round(pancreas_mm3), round(tumor_mm3)])
    print("Finished group:", group)

# Save everything as a table.
with open("outputs/dataset_stats.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "group", "slices", "slice_thickness_mm", "pancreas_mm3", "tumor_mm3"])
    writer.writerows(rows)

# Sort the tumor sizes from small to big.
tumors = sorted(row[5] for row in rows)
print("Patients measured:", len(rows))
print("Patients with no tumor:", sum(1 for t in tumors if t == 0))
print("Smallest tumor (mm3):", tumors[0])                    # The first one is the smallest.
print("Middle tumor (mm3):  ", tumors[len(tumors) // 2])     # The one in the middle.
print("Largest tumor (mm3): ", tumors[-1])                   # The last one is the biggest.

# Count the small tumors. A ball 2 cm wide is 4189 mm3.
print("Tumors smaller than a 2 cm ball:", sum(1 for t in tumors if t < 4189))
for group in ["train", "val", "test"]:
    sizes = [row[5] for row in rows if row[1] == group]
    print("  in", group + ":", sum(1 for t in sizes if t < 4189), "of", len(sizes))

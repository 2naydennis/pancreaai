# dataset_stats.py
# Measures the pancreas and tumor volume of all 281 patients.
# Run: python scripts/dataset_stats.py
# Saves: outputs/dataset_stats.csv
# This is how we found that 110 tumors are smaller than a 2 cm ball.

import csv
import json
import SimpleITK as sitk

folder = "data/Task07_Pancreas/labelsTr"

# 1. Load the three patient groups you saved earlier.
# splits["train"] is then the list of training file names, and so on.
with open("splits.json") as f:
    splits = json.load(f)

# 2. Open every patient's mask and measure it.
# The outer loop goes through the three groups, the inner loop through each patient in a group.
rows = []
for group in ["train", "val", "test"]:
    for name in splits[group]:
        mask = sitk.ReadImage(folder + "/" + name)

        # The volume of one voxel in mm3 is its width * height * depth.
        sx, sy, sz = mask.GetSpacing()
        one_voxel = sx * sy * sz
        grid = sitk.GetArrayFromImage(mask)

        # Count the voxels marked 1 (pancreas) and 2 (tumor), then multiply by the voxel volume.
        pancreas_mm3 = (grid == 1).sum() * one_voxel
        tumor_mm3 = (grid == 2).sum() * one_voxel

        # Add one row for this patient: name, group, slices, slice thickness, pancreas, tumor.
        rows.append([name, group, mask.GetSize()[2], round(sz, 2), round(pancreas_mm3), round(tumor_mm3)])
    print("Finished group:", group)

# 3. Save everything as a table you can open in Excel or Numbers.
# The first row holds the column names, then one row per patient follows.
with open("outputs/dataset_stats.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "group", "slices", "slice_thickness_mm", "pancreas_mm3", "tumor_mm3"])
    writer.writerows(rows)

# 4. Print a summary of tumor sizes.
# row[5] is the tumor volume of a row. sorted puts all of them in order, smallest first.
tumors = sorted(row[5] for row in rows)
print("Patients measured:", len(rows))
print("Patients with no tumor:", sum(1 for t in tumors if t == 0))

# In a sorted list the first item is the smallest, the middle one the median, the last the largest.
print("Smallest tumor (mm3):", tumors[0])
print("Middle tumor (mm3):  ", tumors[len(tumors) // 2])
print("Largest tumor (mm3): ", tumors[-1])

# 5. Count small tumors. A ball 2 cm across has a volume of 4189 mm3.
# First the total, then the same count for each group separately.
print("Tumors smaller than a 2 cm ball:", sum(1 for t in tumors if t < 4189))
for group in ["train", "val", "test"]:
    sizes = [row[5] for row in rows if row[1] == group]
    print("  in", group + ":", sum(1 for t in sizes if t < 4189), "of", len(sizes))

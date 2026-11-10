import csv
import json
import SimpleITK as sitk

folder = "data/Task07_Pancreas/labelsTr"

# 1. Load the three groups you saved earlier
with open("splits.json") as f:
    splits = json.load(f)

# 2. Open every patient's mask and measure it
rows = []
for group in ["train", "val", "test"]:
    for name in splits[group]:
        mask = sitk.ReadImage(folder + "/" + name)
        sx, sy, sz = mask.GetSpacing()
        one_voxel = sx * sy * sz
        grid = sitk.GetArrayFromImage(mask)
        pancreas_mm3 = (grid == 1).sum() * one_voxel
        tumor_mm3 = (grid == 2).sum() * one_voxel
        rows.append([name, group, mask.GetSize()[2], round(sz, 2), round(pancreas_mm3), round(tumor_mm3)])
    print("Finished group:", group)

# 3. Save everything as a table you can open in Excel or Numbers
with open("dataset_stats.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "group", "slices", "slice_thickness_mm", "pancreas_mm3", "tumor_mm3"])
    writer.writerows(rows)

# 4. Print a summary of tumor sizes
tumors = sorted(row[5] for row in rows)
print("Patients measured:", len(rows))
print("Patients with no tumor:", sum(1 for t in tumors if t == 0))
print("Smallest tumor (mm3):", tumors[0])
print("Middle tumor (mm3):  ", tumors[len(tumors) // 2])
print("Largest tumor (mm3): ", tumors[-1])

# 5. Count small tumors. A ball 2 cm across has a volume of 4189 mm3
print("Tumors smaller than a 2 cm ball:", sum(1 for t in tumors if t < 4189))
for group in ["train", "val", "test"]:
    sizes = [row[5] for row in rows if row[1] == group]
    print("  in", group + ":", sum(1 for t in sizes if t < 4189), "of", len(sizes))

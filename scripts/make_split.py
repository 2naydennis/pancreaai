# make_split.py
# Divides the 281 patients into train (197), validation (42) and test (42).
# Run: python scripts/make_split.py   (already done, do not rerun)
# Saves: splits.json
# The model learns from train, we compare experiments on validation, and test is the final exam.
# A patient must never be in two groups, or the model could score well by remembering them.

import os
import json
import random

folder = "data/Task07_Pancreas/imagesTr"

# 1. List every scan file in the folder, in alphabetical order.
# The "if" part keeps only names ending in .nii.gz and skips hidden files that start with a dot.
names = sorted(f for f in os.listdir(folder) if f.endswith(".nii.gz") and not f.startswith("."))
print("Total patients:", len(names))

# 2. Shuffle the list. The seed (42) makes the shuffle come out the same every time,
# so anyone can repeat our split exactly.
random.Random(42).shuffle(names)

# 3. Cut the shuffled list into three groups.
# names[:197] is the first 197 items, names[197:239] the next 42, names[239:] the rest.
train = names[:197]
val = names[197:239]
test = names[239:]

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# 4. Safety check: no patient may appear in two groups.
# & finds names that two groups share, | joins the results. The total must be 0.
overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
print("Patients in more than one group:", len(overlap))

# 5. Save the groups to a file, so every later script uses the same split.
# "with open(...) as f" opens the file, and closes it again when the indented part ends.
with open("splits.json", "w") as f:
    json.dump({"train": train, "val": val, "test": test}, f, indent=2)
print("Saved splits.json")

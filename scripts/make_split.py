# make_split.py: splits the 281 patients into three groups.
# Train is for learning, val is the practice test, test is the final test.
# Run: python scripts/make_split.py   (already done, do not run again)

import os
import json
import random

folder = "data/Task07_Pancreas/imagesTr"

# Make a list of all the scan files.
names = sorted(f for f in os.listdir(folder) if f.endswith(".nii.gz") and not f.startswith("."))
print("Total patients:", len(names))

# Shuffle the list. The number 42 makes it shuffle the same way every time.
random.Random(42).shuffle(names)

# Cut the list into three groups.
train = names[:197]        # The first 197 patients.
val = names[197:239]       # The next 42.
test = names[239:]         # The last 42.

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# Check that no patient is in two groups. This must print 0.
overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
print("Patients in more than one group:", len(overlap))

# Save the groups, so every other script uses the same ones.
with open("splits.json", "w") as f:
    json.dump({"train": train, "val": val, "test": test}, f, indent=2)
print("Saved splits.json")

import os
import json
import random

folder = "data/Task07_Pancreas/imagesTr"

# 1. List every scan file (skip hidden files that start with a dot)
names = sorted(f for f in os.listdir(folder) if f.endswith(".nii.gz") and not f.startswith("."))
print("Total patients:", len(names))

# 2. Shuffle them. The seed (42) makes the shuffle come out the same every time
random.Random(42).shuffle(names)

# 3. Cut the shuffled list into three groups
train = names[:197]
val = names[197:239]
test = names[239:]

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# 4. Safety check: no patient may appear in two groups
overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
print("Patients in more than one group:", len(overlap))

# 5. Save the groups to a file, so every later script uses the same split
with open("splits.json", "w") as f:
    json.dump({"train": train, "val": val, "test": test}, f, indent=2)
print("Saved splits.json")

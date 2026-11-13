# threshold_test.py
# Experiment 2: does a lower or higher decision bar improve the score?
# The bar (threshold) is how sure the model must be before a pixel counts as pancreas.
# Run: python scripts/threshold_test.py
# Saves: outputs/threshold_results.csv
# Result: the score barely moved (0.640 at 0.1, 0.639 at 0.5), so the bar is not the problem.

import csv
import json
import numpy as np
import torch
import SimpleITK as sitk
from unet import UNet

folder = "data/Task07_Pancreas"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# The bars to try. 0.5 is what the model uses now.
bars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Load the baseline model and switch it to exam mode (no learning).
model = UNet().to(device)
model.load_state_dict(torch.load("models/unet_small.pt", map_location=device))
model.eval()

with open("splits.json") as f:
    splits = json.load(f)

# Two sets of labelled boxes, one box per bar.
# scores collects the patient scores, coloured counts the voxels the model coloured.
scores = {bar: [] for bar in bars}
coloured = {bar: 0 for bar in bars}

# The number of voxels the doctors coloured, added up over all patients.
true_total = 0

for name in splits["val"]:
    # Load this patient and prepare every slice: window, normalise, shrink.
    scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
    mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))
    slices = (scan_grid.clip(-100, 240) + 100) / 340
    slices = torch.from_numpy(slices[:, ::2, ::2].astype(np.float32)).unsqueeze(1)
    truth = mask_grid[:, ::2, ::2] > 0
    true_total += truth.sum()

    # 1. Get "how sure" (0..1) for every pixel, once.
    # The model is the slow part, so we run it one time and reuse the answer for all seven bars.
    sure = []
    with torch.no_grad():
        for i in range(0, len(slices), 16):
            sure.append(torch.sigmoid(model(slices[i:i + 16].to(device)))[:, 0].cpu().numpy())
    sure = np.concatenate(sure)

    # 2. Turn "how sure" into a mask using each bar, and score it.
    # A voxel counts as pancreas when the model is more sure than the bar.
    for bar in bars:
        guess = sure > bar
        dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())
        scores[bar].append(dice)
        coloured[bar] += guess.sum()

    print("done", name, flush=True)

# 3. One row per bar, printed in the terminal and saved to the file.
print()
print("bar   average Dice   patients below 0.5   coloured vs doctor")
with open("outputs/threshold_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["bar", "average_dice", "patients_below_0.5", "coloured_vs_doctor_percent"])
    for bar in bars:
        # The average score of the 42 patients, and how many of them scored under 0.5.
        average = sum(scores[bar]) / len(scores[bar])
        failing = sum(1 for s in scores[bar] if s < 0.5)

        # What the model coloured as a percentage of what the doctors coloured.
        # Under 100 means the model draws too little.
        percent = 100 * coloured[bar] / true_total
        print(bar, "     ", round(float(average), 3), "           ", failing, "                 ", round(float(percent)), "%")
        writer.writerow([bar, round(float(average), 4), failing, round(float(percent), 1)])

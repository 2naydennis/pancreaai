# threshold_test.py: experiment 2. Tries different "how sure" bars for calling a pixel pancreas.
# Run: python scripts/threshold_test.py
# Saves: outputs/threshold_results.csv
# Result: the bar was not the problem.

import csv
import json
import numpy as np
import torch
import SimpleITK as sitk
from unet import UNet

folder = "data/Task07_Pancreas"
device = "mps" if torch.backends.mps.is_available() else "cpu"

bars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]    # The bars to try. The normal one is 0.5.

# Load the model.
model = UNet().to(device)
model.load_state_dict(torch.load("models/unet_small.pt", map_location=device))
model.eval()

with open("splits.json") as f:
    splits = json.load(f)

scores = {bar: [] for bar in bars}     # The scores for each bar.
coloured = {bar: 0 for bar in bars}    # How many pixels the model coloured with each bar.
true_total = 0                         # How many pixels the doctor coloured.

for name in splits["val"]:
    # Load the patient and get every slice ready.
    scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
    mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))
    slices = (scan_grid.clip(-100, 240) + 100) / 340
    slices = torch.from_numpy(slices[:, ::2, ::2].astype(np.float32)).unsqueeze(1)
    truth = mask_grid[:, ::2, ::2] > 0
    true_total += truth.sum()

    # Ask the model "how sure are you?" only once, to save time.
    sure = []
    with torch.no_grad():
        for i in range(0, len(slices), 16):
            sure.append(torch.sigmoid(model(slices[i:i + 16].to(device)))[:, 0].cpu().numpy())
    sure = np.concatenate(sure)

    # Score the patient with each bar.
    for bar in bars:
        guess = sure > bar
        dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())
        scores[bar].append(dice)
        coloured[bar] += guess.sum()

    print("done", name, flush=True)

# Print and save the table.
print()
print("bar   average Dice   patients below 0.5   coloured vs doctor")
with open("outputs/threshold_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["bar", "average_dice", "patients_below_0.5", "coloured_vs_doctor_percent"])
    for bar in bars:
        average = sum(scores[bar]) / len(scores[bar])
        failing = sum(1 for s in scores[bar] if s < 0.5)    # Patients with a bad score.
        percent = 100 * coloured[bar] / true_total          # Under 100 means the model draws too little.
        print(bar, "     ", round(float(average), 3), "           ", failing, "                 ", round(float(percent)), "%")
        writer.writerow([bar, round(float(average), 4), failing, round(float(percent), 1)])

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

model = UNet().to(device)
model.load_state_dict(torch.load("models/unet_best.pt", map_location=device))
model.eval()

with open("splits.json") as f:
    splits = json.load(f)

# One list of patient scores for each bar
scores = {bar: [] for bar in bars}
coloured = {bar: 0 for bar in bars}
true_total = 0

for name in splits["val"]:
    scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
    mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))
    slices = (scan_grid.clip(-100, 240) + 100) / 340
    slices = torch.from_numpy(slices[:, ::2, ::2].astype(np.float32)).unsqueeze(1)
    truth = mask_grid[:, ::2, ::2] > 0
    true_total += truth.sum()

    # 1. Get "how sure" (0..1) for every pixel, once
    sure = []
    with torch.no_grad():
        for i in range(0, len(slices), 16):
            sure.append(torch.sigmoid(model(slices[i:i + 16].to(device)))[:, 0].cpu().numpy())
    sure = np.concatenate(sure)

    # 2. Turn "how sure" into a mask using each bar, and score it
    for bar in bars:
        guess = sure > bar
        dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())
        scores[bar].append(dice)
        coloured[bar] += guess.sum()

    print("done", name, flush=True)

# 3. One row per bar
print()
print("bar   average Dice   patients below 0.5   coloured vs doctor")
with open("outputs/threshold_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["bar", "average_dice", "patients_below_0.5", "coloured_vs_doctor_percent"])
    for bar in bars:
        average = sum(scores[bar]) / len(scores[bar])
        failing = sum(1 for s in scores[bar] if s < 0.5)
        percent = 100 * coloured[bar] / true_total
        print(bar, "     ", round(float(average), 3), "           ", failing, "                 ", round(float(percent)), "%")
        writer.writerow([bar, round(float(average), 4), failing, round(float(percent), 1)])

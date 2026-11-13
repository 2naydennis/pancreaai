import csv
import json
import numpy as np
import torch
import SimpleITK as sitk
from unet import UNet

folder = "data/Task07_Pancreas"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. Load your trained model
model = UNet().to(device)
model.load_state_dict(torch.load("models/unet_best.pt", map_location=device))
model.eval()

with open("splits.json") as f:
    splits = json.load(f)

rows = []
for name in splits["val"]:
    scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
    mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

    # 2. Prepare EVERY slice of this patient: window, normalise, shrink
    slices = (scan_grid.clip(-100, 240) + 100) / 340
    slices = torch.from_numpy(slices[:, ::2, ::2].astype(np.float32)).unsqueeze(1)
    truth = mask_grid[:, ::2, ::2] > 0

    # 3. Ask the model for a mask on every slice, 16 slices at a time
    guess = []
    with torch.no_grad():
        for i in range(0, len(slices), 16):
            out = torch.sigmoid(model(slices[i:i + 16].to(device))) > 0.5
            guess.append(out[:, 0].cpu().numpy())
    guess = np.concatenate(guess)

    # 4. One Dice score for the whole 3D patient
    overlap = (guess & truth).sum()
    dice = 2 * overlap / (guess.sum() + truth.sum())

    # 5. How much did the model colour in slices that have no pancreas at all?
    empty = ~truth.any(axis=(1, 2))
    false_alarm = guess[empty].sum()

    rows.append([name, round(float(dice), 3), int(truth.sum()), int(guess.sum()), int(false_alarm)])
    print(name, " Dice", round(float(dice), 3), flush=True)

# 6. Save one row per patient
with open("outputs/val_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "dice", "true_voxels", "guessed_voxels", "false_alarm_voxels"])
    writer.writerows(rows)

# 7. Summary
scores = sorted(row[1] for row in rows)
print("Patients scored:", len(scores))
print("Average Dice:", round(sum(scores) / len(scores), 3))
print("Middle Dice: ", scores[len(scores) // 2])
print("Worst Dice:  ", scores[0])
print("Best Dice:   ", scores[-1])
print("Patients below 0.5:", sum(1 for s in scores if s < 0.5))

# evaluate.py: the real exam. It scores each patient as a whole 3D scan.
# Run: python scripts/evaluate.py NAME WIDTH       (example: unet_small 16)
#      python scripts/evaluate.py NAME WIDTH 3     (example: unet_3slice 16 3)
# Saves: outputs/val_results_NAME.csv

import csv
import sys
import json
import numpy as np
import torch
import SimpleITK as sitk
from unet import UNet

NAME = sys.argv[1]          # The model's name.
WIDTH = int(sys.argv[2])    # 16 for small, 32 for big.
SLICES = int(sys.argv[3]) if len(sys.argv) > 3 else 1    # 1 or 3.

folder = "data/Task07_Pancreas"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load the trained model.
model = UNet(width=WIDTH, slices=SLICES).to(device)
model.load_state_dict(torch.load("models/" + NAME + ".pt", map_location=device))
model.eval()    # Exam mode: the model answers but does not learn.

with open("splits.json") as f:
    splits = json.load(f)

rows = []
for name in splits["val"]:    # Go through each validation patient.
    scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
    mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

    # Get every slice ready, the same way as in training.
    slices = (scan_grid.clip(-100, 240) + 100) / 340
    slices = torch.from_numpy(slices[:, ::2, ::2].astype(np.float32)).unsqueeze(1)
    truth = mask_grid[:, ::2, ::2] > 0    # The doctor's answer.

    # For the 3-slice model, glue each slice to the one below and the one above.
    if SLICES == 3:
        below = torch.cat([slices[:1], slices[:-1]])
        above = torch.cat([slices[1:], slices[-1:]])
        slices = torch.cat([below, slices, above], dim=1)

    # The model guesses, 16 slices at a time.
    guess = []
    with torch.no_grad():
        for i in range(0, len(slices), 16):
            out = torch.sigmoid(model(slices[i:i + 16].to(device))) > 0.5    # More than 0.5 sure means pancreas.
            guess.append(out[:, 0].cpu().numpy())
    guess = np.concatenate(guess)    # Put the slices back together into one 3D scan.

    # Give the whole patient one score.
    overlap = (guess & truth).sum()    # Where the model and the doctor both say pancreas.
    dice = 2 * overlap / (guess.sum() + truth.sum())

    # Count what the model coloured on slices that have no pancreas at all.
    empty = ~truth.any(axis=(1, 2))
    false_alarm = guess[empty].sum()

    rows.append([name, round(float(dice), 3), int(truth.sum()), int(guess.sum()), int(false_alarm)])
    print(name, " Dice", round(float(dice), 3), flush=True)

# Save one row per patient.
with open("outputs/val_results_" + NAME + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "dice", "true_voxels", "guessed_voxels", "false_alarm_voxels"])
    writer.writerows(rows)

# Print a summary, with the scores sorted from low to high.
scores = sorted(row[1] for row in rows)
print("Model:", NAME)
print("Patients scored:", len(scores))
print("Average Dice:", round(sum(scores) / len(scores), 3))
print("Middle Dice: ", scores[len(scores) // 2])
print("Worst Dice:  ", scores[0])
print("Best Dice:   ", scores[-1])
print("Patients below 0.5:", sum(1 for s in scores if s < 0.5))

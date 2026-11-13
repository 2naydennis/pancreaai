# evaluate.py
# The real exam: scores each of the 42 validation patients in full 3D.
# Run: python scripts/evaluate.py NAME WIDTH        example: unet_small 16
#      python scripts/evaluate.py NAME WIDTH 3      example: unet_3slice 16 3
# Saves: outputs/val_results_NAME.csv
# train.py scores a selection of slices. This script scores whole patients, every slice included,
# so it is the number we report.

import csv
import sys
import json
import numpy as np
import torch
import SimpleITK as sitk
from unet import UNet

# Which model to score comes from the words you type after the script name.
NAME = sys.argv[1]          # the model's name, as used in train.py
WIDTH = int(sys.argv[2])    # its size: 16 = small, 32 = big
SLICES = int(sys.argv[3]) if len(sys.argv) > 3 else 1    # slices the model sees at once: 1 or 3

folder = "data/Task07_Pancreas"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. Load your trained model.
# Build an empty model of the same size, then pour the saved weights into it.
model = UNet(width=WIDTH, slices=SLICES).to(device)
model.load_state_dict(torch.load("models/" + NAME + ".pt", map_location=device))
model.eval()

with open("splits.json") as f:
    splits = json.load(f)

# The loop runs once per validation patient and adds one row of results each time.
rows = []
for name in splits["val"]:
    scan_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/imagesTr/" + name))
    mask_grid = sitk.GetArrayFromImage(sitk.ReadImage(folder + "/labelsTr/" + name))

    # 2. Prepare EVERY slice of this patient: window, normalise, shrink.
    # This must match the training preparation exactly, or the model will fail.
    slices = (scan_grid.clip(-100, 240) + 100) / 340
    slices = torch.from_numpy(slices[:, ::2, ::2].astype(np.float32)).unsqueeze(1)

    # The doctor's answer, shrunk the same way. True means pancreas or tumor.
    truth = mask_grid[:, ::2, ::2] > 0

    # For a 3-slice model: glue each slice together with the one below and the one above.
    # The first and last slice have no neighbour on one side, so they are repeated.
    if SLICES == 3:
        below = torch.cat([slices[:1], slices[:-1]])
        above = torch.cat([slices[1:], slices[-1:]])
        slices = torch.cat([below, slices, above], dim=1)

    # 3. Ask the model for a mask on every slice, 16 slices at a time.
    # A voxel counts as pancreas when the model is more than 0.5 sure.
    guess = []
    with torch.no_grad():
        for i in range(0, len(slices), 16):
            out = torch.sigmoid(model(slices[i:i + 16].to(device))) > 0.5
            guess.append(out[:, 0].cpu().numpy())

    # Join the batches into one 3D mask with the same shape as the doctor's.
    guess = np.concatenate(guess)

    # 4. One Dice score for the whole 3D patient.
    # overlap counts the voxels where model and doctor both say pancreas.
    overlap = (guess & truth).sum()
    dice = 2 * overlap / (guess.sum() + truth.sum())

    # 5. How much did the model colour in slices that have no pancreas at all?
    # empty is True for those slices. Anything coloured there is a false alarm.
    empty = ~truth.any(axis=(1, 2))
    false_alarm = guess[empty].sum()

    rows.append([name, round(float(dice), 3), int(truth.sum()), int(guess.sum()), int(false_alarm)])
    print(name, " Dice", round(float(dice), 3), flush=True)

# 6. Save one row per patient, under a header row with the column names.
with open("outputs/val_results_" + NAME + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "dice", "true_voxels", "guessed_voxels", "false_alarm_voxels"])
    writer.writerows(rows)

# 7. Summary.
# row[1] is a patient's Dice score. Sorted from lowest to highest, the first is the worst,
# the middle one the median, and the last the best.
scores = sorted(row[1] for row in rows)
print("Model:", NAME)
print("Patients scored:", len(scores))
print("Average Dice:", round(sum(scores) / len(scores), 3))
print("Middle Dice: ", scores[len(scores) // 2])
print("Worst Dice:  ", scores[0])
print("Best Dice:   ", scores[-1])
print("Patients below 0.5:", sum(1 for s in scores if s < 0.5))

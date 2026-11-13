# show_failures.py
# Experiment 4: draws the six patients that failed with both models.
# Run: python scripts/show_failures.py
# Saves: outputs/failures.png
# Result: in most of them the model finds one piece of the pancreas and misses the rest.

import numpy as np
import torch
import matplotlib.pyplot as plt
from unet import UNet

# The six patients that failed with BOTH models (Dice below 0.5).
failures = ["pancreas_099", "pancreas_037", "pancreas_414",
            "pancreas_058", "pancreas_347", "pancreas_357"]

# 1. Load the small model (the baseline) and switch it to exam mode (no learning).
model = UNet(width=16)
model.load_state_dict(torch.load("models/unet_small.pt", map_location="cpu"))
model.eval()

# 2. Load the validation slices, their masks, and the patient each slice came from.
val = np.load("data/slices/val.npz")
images, masks, patients = val["images"], val["masks"], val["patients"]

# Two rows of three pictures. zip pairs each picture area with one patient name.
fig, panels = plt.subplots(2, 3, figsize=(16, 11))
for panel, name in zip(panels.flat, failures):
    # 3. For this patient, take the slice with the most pancreas.
    # The file stores full names, so ".nii.gz" is glued on before comparing.
    mine = np.where(patients == name + ".nii.gz")[0]
    best = mine[masks[mine].sum(axis=(1, 2)).argmax()]

    # 4. Ask the model for its mask on that slice.
    # A pixel counts as pancreas when the model is more than 0.5 sure.
    x = torch.from_numpy(images[best]).float().div(255).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        guess = (torch.sigmoid(model(x)) > 0.5)[0, 0].numpy()
    truth = masks[best] > 0

    # Dice for this one slice: 2 * overlap / (model's area + doctor's area).
    dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())
    print(name, " Dice on this slice:", round(float(dice), 2))

    # 5. Draw: doctor in green, model in red.
    # [::-1, ::-1] turns the picture 180 degrees, the way doctors view a CT.
    panel.imshow(images[best][::-1, ::-1], cmap="gray")
    panel.contour(truth[::-1, ::-1], colors="lime", linewidths=1.5)
    if guess.any():
        panel.contour(guess[::-1, ::-1], colors="red", linewidths=1.5)
    panel.set_title(name + "   Dice " + str(round(float(dice), 2)))
    panel.axis("off")

fig.suptitle("The six shared failures. Green = doctor, red = model")
plt.savefig("outputs/failures.png", dpi=100, bbox_inches="tight")
print("Saved outputs/failures.png")

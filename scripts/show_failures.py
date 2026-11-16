# show_failures.py: experiment 4. Draws the 6 patients that both models failed on.
# Run: python scripts/show_failures.py
# Saves: outputs/failures.png
# Result: the model finds one piece of the pancreas and misses the rest.

import numpy as np
import torch
import matplotlib.pyplot as plt
from unet import UNet

# The 6 patients that both models failed on.
failures = ["pancreas_099", "pancreas_037", "pancreas_414",
            "pancreas_058", "pancreas_347", "pancreas_357"]

# Load the small model.
model = UNet(width=16)
model.load_state_dict(torch.load("models/unet_small.pt", map_location="cpu"))
model.eval()

# Load the validation slices.
val = np.load("data/slices/val.npz")
images, masks, patients = val["images"], val["masks"], val["patients"]

fig, panels = plt.subplots(2, 3, figsize=(16, 11))    # 2 rows of 3 pictures.
for panel, name in zip(panels.flat, failures):
    # Find the slice with the most pancreas for this patient.
    mine = np.where(patients == name + ".nii.gz")[0]
    best = mine[masks[mine].sum(axis=(1, 2)).argmax()]

    # Ask the model for its guess.
    x = torch.from_numpy(images[best]).float().div(255).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        guess = (torch.sigmoid(model(x)) > 0.5)[0, 0].numpy()
    truth = masks[best] > 0

    dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())    # Score this slice.
    print(name, " Dice on this slice:", round(float(dice), 2))

    # Draw it: green is the doctor, red is the model.
    panel.imshow(images[best][::-1, ::-1], cmap="gray")
    panel.contour(truth[::-1, ::-1], colors="lime", linewidths=1.5)
    if guess.any():
        panel.contour(guess[::-1, ::-1], colors="red", linewidths=1.5)
    panel.set_title(name + "   Dice " + str(round(float(dice), 2)))
    panel.axis("off")

fig.suptitle("The six shared failures. Green = doctor, red = model")
plt.savefig("outputs/failures.png", dpi=100, bbox_inches="tight")
print("Saved outputs/failures.png")

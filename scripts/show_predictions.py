# show_predictions.py: draws the doctor's outline (green) and the model's (red) on 4 patients.
# Run: python scripts/show_predictions.py
# Saves: outputs/predictions.png

import numpy as np
import torch
import matplotlib.pyplot as plt
from unet import UNet

# Load the trained model.
model = UNet()
model.load_state_dict(torch.load("models/unet_small.pt", map_location="cpu"))
model.eval()    # Exam mode: the model answers but does not learn.

# Load the validation slices.
val = np.load("data/slices/val.npz")
images, masks, patients = val["images"], val["masks"], val["patients"]

# Pick the first 4 patients.
names = sorted(set(patients.tolist()))[:4]

fig, panels = plt.subplots(1, 4, figsize=(20, 5.5))    # 4 pictures in a row.
for panel, name in zip(panels, names):
    mine = np.where(patients == name)[0]                  # Find this patient's slices.
    best = mine[masks[mine].sum(axis=(1, 2)).argmax()]    # Pick the one with the most pancreas.

    # Ask the model for its guess.
    x = torch.from_numpy(images[best]).float().div(255).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        guess = (torch.sigmoid(model(x)) > 0.5)[0, 0].numpy()    # More than 0.5 sure means pancreas.
    truth = masks[best] > 0

    # Score this one slice.
    dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())
    print(name, " Dice on this slice:", round(float(dice), 2))

    # Draw it: green is the doctor, red is the model.
    panel.imshow(images[best][::-1, ::-1], cmap="gray")
    panel.contour(truth[::-1, ::-1], colors="lime", linewidths=1.5)
    if guess.any():    # Only draw red if the model coloured something.
        panel.contour(guess[::-1, ::-1], colors="red", linewidths=1.5)
    panel.set_title(name.replace(".nii.gz", "") + "   Dice " + str(round(float(dice), 2)))
    panel.axis("off")

fig.suptitle("Green = doctor, red = your model")
plt.savefig("outputs/predictions.png", dpi=100, bbox_inches="tight")
print("Saved picture")

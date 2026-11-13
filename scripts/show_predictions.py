# show_predictions.py
# Draws the model's outline (red) next to the doctor's (green) on four validation patients.
# Run: python scripts/show_predictions.py
# Saves: outputs/predictions.png
# A score tells you how good the model is. A picture shows you what it gets wrong.

import numpy as np
import torch
import matplotlib.pyplot as plt
from unet import UNet

# 1. Load your trained model.
# Build an empty model, pour the saved weights into it, and switch to exam mode (no learning).
model = UNet()
model.load_state_dict(torch.load("models/unet_small.pt", map_location="cpu"))
model.eval()

# 2. Load validation slices: patients the model never trained on.
# images[k] is a slice, masks[k] the doctor's mask for it, patients[k] the patient it came from.
val = np.load("data/slices/val.npz")
images, masks, patients = val["images"], val["masks"], val["patients"]

# 3. Take four different patients, and for each the slice with the most pancreas.
# set(...) removes repeated names, sorted puts them in order, [:4] keeps the first four.
names = sorted(set(patients.tolist()))[:4]

# One row of four pictures. zip pairs each picture area with one patient name.
fig, panels = plt.subplots(1, 4, figsize=(20, 5.5))
for panel, name in zip(panels, names):
    # mine = the positions of this patient's slices. best = the one with the most pancreas pixels.
    mine = np.where(patients == name)[0]
    best = mine[masks[mine].sum(axis=(1, 2)).argmax()]

    # 4. Ask the model for its mask.
    # First turn the slice into what the model expects: numbers 0..1 in the shape (1, 1, 256, 256).
    x = torch.from_numpy(images[best]).float().div(255).unsqueeze(0).unsqueeze(0)

    # A pixel counts as pancreas when the model is more than 0.5 sure.
    with torch.no_grad():
        guess = (torch.sigmoid(model(x)) > 0.5)[0, 0].numpy()
    truth = masks[best] > 0

    # 5. Score this one slice with Dice: 2 * overlap / (model's area + doctor's area).
    dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())
    print(name, " Dice on this slice:", round(float(dice), 2))

    # 6. Draw: doctor in green, model in red.
    # [::-1, ::-1] turns the picture 180 degrees, the way doctors view a CT.
    panel.imshow(images[best][::-1, ::-1], cmap="gray")
    panel.contour(truth[::-1, ::-1], colors="lime", linewidths=1.5)

    # Draw the red line only if the model coloured anything at all.
    if guess.any():
        panel.contour(guess[::-1, ::-1], colors="red", linewidths=1.5)
    panel.set_title(name.replace(".nii.gz", "") + "   Dice " + str(round(float(dice), 2)))
    panel.axis("off")

fig.suptitle("Green = doctor, red = your model")
plt.savefig("outputs/predictions.png", dpi=100, bbox_inches="tight")
print("Saved picture")

import numpy as np
import torch
import matplotlib.pyplot as plt
from unet import UNet

# The six patients that failed with BOTH models
failures = ["pancreas_099", "pancreas_037", "pancreas_414",
            "pancreas_058", "pancreas_347", "pancreas_357"]

# 1. Load the small model (the baseline)
model = UNet(width=16)
model.load_state_dict(torch.load("models/unet_small.pt", map_location="cpu"))
model.eval()

# 2. Load the validation slices
val = np.load("data/slices/val.npz")
images, masks, patients = val["images"], val["masks"], val["patients"]

fig, panels = plt.subplots(2, 3, figsize=(16, 11))
for panel, name in zip(panels.flat, failures):
    # 3. For this patient, take the slice with the most pancreas
    mine = np.where(patients == name + ".nii.gz")[0]
    best = mine[masks[mine].sum(axis=(1, 2)).argmax()]

    # 4. Ask the model for its mask on that slice
    x = torch.from_numpy(images[best]).float().div(255).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        guess = (torch.sigmoid(model(x)) > 0.5)[0, 0].numpy()
    truth = masks[best] > 0

    dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())
    print(name, " Dice on this slice:", round(float(dice), 2))

    # 5. Draw: doctor in green, model in red
    panel.imshow(images[best][::-1, ::-1], cmap="gray")
    panel.contour(truth[::-1, ::-1], colors="lime", linewidths=1.5)
    if guess.any():
        panel.contour(guess[::-1, ::-1], colors="red", linewidths=1.5)
    panel.set_title(name + "   Dice " + str(round(float(dice), 2)))
    panel.axis("off")

fig.suptitle("The six shared failures. Green = doctor, red = model")
plt.savefig("outputs/failures.png", dpi=100, bbox_inches="tight")
print("Saved outputs/failures.png")

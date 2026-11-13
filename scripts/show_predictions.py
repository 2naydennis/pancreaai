import numpy as np
import torch
import matplotlib.pyplot as plt
from unet import UNet

# 1. Load your trained model
model = UNet()
model.load_state_dict(torch.load("models/unet_small.pt", map_location="cpu"))
model.eval()

# 2. Load validation slices: patients the model never trained on
val = np.load("data/slices/val.npz")
images, masks, patients = val["images"], val["masks"], val["patients"]

# 3. Take four different patients, and for each the slice with the most pancreas
names = sorted(set(patients.tolist()))[:4]

fig, panels = plt.subplots(1, 4, figsize=(20, 5.5))
for panel, name in zip(panels, names):
    mine = np.where(patients == name)[0]
    best = mine[masks[mine].sum(axis=(1, 2)).argmax()]

    # 4. Ask the model for its mask
    x = torch.from_numpy(images[best]).float().div(255).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        guess = (torch.sigmoid(model(x)) > 0.5)[0, 0].numpy()
    truth = masks[best] > 0

    # 5. Score this one slice with Dice
    dice = 2 * (guess & truth).sum() / (guess.sum() + truth.sum())
    print(name, " Dice on this slice:", round(float(dice), 2))

    # 6. Draw: doctor in green, model in red
    panel.imshow(images[best][::-1, ::-1], cmap="gray")
    panel.contour(truth[::-1, ::-1], colors="lime", linewidths=1.5)
    if guess.any():
        panel.contour(guess[::-1, ::-1], colors="red", linewidths=1.5)
    panel.set_title(name.replace(".nii.gz", "") + "   Dice " + str(round(float(dice), 2)))
    panel.axis("off")

fig.suptitle("Green = doctor, red = your model")
plt.savefig("outputs/predictions.png", dpi=100, bbox_inches="tight")
print("Saved picture")

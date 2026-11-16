# train.py: teaches the model and keeps the best version.
# Run: python scripts/train.py NAME WIDTH       (example: unet_small 16)
#      python scripts/train.py NAME WIDTH 3     (example: unet_3slice 16 3)
# Saves: models/NAME.pt and outputs/train_log_NAME.csv
# An epoch is one pass over all slices. Loss is how wrong the model is. Dice is the score from 0 to 1.

import os
import sys
import time
import numpy as np
import torch
from unet import UNet    # Our model from unet.py.

# Read the words you typed after the script name.
NAME = sys.argv[1]          # The model's name.
WIDTH = int(sys.argv[2])    # 16 for small, 32 for big.
SLICES = int(sys.argv[3]) if len(sys.argv) > 3 else 1    # 1 or 3. If you type nothing, it is 1.

EPOCHS = 5     # Go through all the slices 5 times.
BATCH = 16     # Look at 16 slices at once.

device = "mps" if torch.backends.mps.is_available() else "cpu"    # Use the Mac's fast chip.
print("Training on:", device, flush=True)

# Load the slices. The 3-slice model uses the files with a 3 in the name.
ending = ".npz" if SLICES == 1 else "3.npz"
train = np.load("data/slices/train" + ending)
val = np.load("data/slices/val" + ending)
# x is the picture, y is the right answer.
train_x, train_y = torch.from_numpy(train["images"]), torch.from_numpy(train["masks"])
val_x, val_y = torch.from_numpy(val["images"]), torch.from_numpy(val["masks"])
print("Train slices:", len(train_x), " Val slices:", len(val_x), flush=True)

# Get slices ready: change 0-255 to 0-1 and send them to the fast chip.
def to_model(images, masks):
    if SLICES == 1:
        images = images.unsqueeze(1)     # Add the extra slot the model expects.
    x = images.float().div(255).to(device)
    y = masks.unsqueeze(1).float().to(device)
    return x, y

# The loss measures how wrong the model is, using two checks added together.
pixel_loss = torch.nn.BCEWithLogitsLoss()    # Check 1: is each pixel right?

def loss_fn(guess, truth):
    sure = torch.sigmoid(guess)                 # Turn the model's numbers into "how sure" from 0 to 1.
    overlap = (sure * truth).sum()              # Where the model and the doctor agree.
    dice = (2 * overlap + 1) / (sure.sum() + truth.sum() + 1)    # Check 2: how much do they overlap?
    return pixel_loss(guess, truth) + (1 - dice)

torch.manual_seed(0)    # Start the same way every run, so runs are fair to compare.
model = UNet(width=WIDTH, slices=SLICES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    # Moves the weights. lr is the step size.
print("Model:", NAME, " width:", WIDTH, " slices:", SLICES, " weights:", sum(p.numel() for p in model.parameters()), flush=True)

os.makedirs("models", exist_ok=True)
log = open("outputs/train_log_" + NAME + ".csv", "w")
log.write("epoch,train_loss,val_dice,minutes\n")
best_dice = 0    # The best score so far.

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    # Learning time: the same 4 steps as learn_demo.py.
    model.train()
    order = torch.randperm(len(train_x))    # Shuffle the slices.
    batches = len(order) // BATCH
    loss_sum = 0
    for n in range(batches):
        pick = order[n * BATCH:(n + 1) * BATCH]    # Take the next 16 slices.
        x, y = to_model(train_x[pick], train_y[pick])

        guess = model(x)             # 1. The model makes a guess.
        loss = loss_fn(guess, y)     # 2. Measure how wrong it is.
        optimizer.zero_grad()        #    Forget the old directions.
        loss.backward()              # 3. Work out which way each weight should move.
        optimizer.step()             # 4. Move every weight a small step.

        loss_sum += loss.item()
        if n % 100 == 0:             # Show progress every 100 batches.
            print("  epoch", epoch, " batch", n, "of", batches, " loss", round(loss.item(), 3), flush=True)

    # Exam time on patients the model has never seen. No learning here.
    model.eval()
    overlap = guessed = true = 0
    with torch.no_grad():
        for i in range(0, len(val_x), BATCH):
            x, y = to_model(val_x[i:i + BATCH], val_y[i:i + BATCH])
            mask = (torch.sigmoid(model(x)) > 0.5).float()    # More than 0.5 sure means pancreas.
            overlap += (mask * y).sum().item()
            guessed += mask.sum().item()
            true += y.sum().item()
    val_dice = 2 * overlap / (guessed + true)    # The exam score.

    minutes = (time.time() - start) / 60
    train_loss = loss_sum / batches
    print("EPOCH", epoch, " train loss", round(train_loss, 3), " val Dice", round(val_dice, 3), " minutes", round(minutes, 1), flush=True)
    log.write(str(epoch) + "," + str(round(train_loss, 4)) + "," + str(round(val_dice, 4)) + "," + str(round(minutes, 1)) + "\n")
    log.flush()

    # If this is the best score so far, save the model.
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "models/" + NAME + ".pt")
        print("  saved new best model", flush=True)

log.close()
print("Finished. Best val Dice:", round(best_dice, 3))

# train.py
# Trains a U-Net and saves the best version.
# Run: python scripts/train.py NAME WIDTH        example: unet_small 16
#      python scripts/train.py NAME WIDTH 3      example: unet_3slice 16 3
# Saves: models/NAME.pt and outputs/train_log_NAME.csv
#
# epoch = one full pass over the training slices
# batch = 16 slices the model looks at together
# loss  = how wrong the model is (lower is better)
# Dice  = overlap with the doctor's mask, from 0 to 1 (higher is better)

import os
import sys
import time
import numpy as np
import torch
from unet import UNet

# The settings come from the words you type after the script name.
# sys.argv holds those words as text, and int(...) turns text like "32" into a number.
NAME = sys.argv[1]          # a name for this model, used in the saved file names
WIDTH = int(sys.argv[2])    # model size: 16 = small, 32 = big

# The third word is optional. If you leave it out, SLICES is 1.
SLICES = int(sys.argv[3]) if len(sys.argv) > 3 else 1    # slices the model sees at once: 1 or 3

EPOCHS = 5
BATCH = 16

# Use the Mac's GPU ("mps") if it is available, otherwise the slower CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Training on:", device, flush=True)

# 1. Load the slices you built with make_slices.py (1 slice) or make_slices_3.py (3 slices).
# The file is train.npz for 1 slice and train3.npz for 3 slices.
ending = ".npz" if SLICES == 1 else "3.npz"
train = np.load("data/slices/train" + ending)
val = np.load("data/slices/val" + ending)

# x is what goes into the model (the slices), y is the correct answer (the masks).
train_x, train_y = torch.from_numpy(train["images"]), torch.from_numpy(train["masks"])
val_x, val_y = torch.from_numpy(val["images"]), torch.from_numpy(val["masks"])
print("Train slices:", len(train_x), " Val slices:", len(val_x), flush=True)

# Turn stored slices (0..255) back into what the model expects (0..1) and move them to the GPU.
# unsqueeze(1) adds an empty slot to the shape, because the model wants (how many, slices, 256, 256).
def to_model(images, masks):
    if SLICES == 1:
        images = images.unsqueeze(1)     # a single slice still needs a "how many slices" slot
    x = images.float().div(255).to(device)
    y = masks.unsqueeze(1).float().to(device)
    return x, y

# 2. The loss: how wrong is the model's mask? Lower is better.
# We add two measures. The pixel loss checks every pixel on its own.
# The Dice part checks the overlap, so the huge background cannot hide a missed pancreas.
pixel_loss = torch.nn.BCEWithLogitsLoss()

def loss_fn(guess, truth):
    sure = torch.sigmoid(guess)                 # turn raw numbers into 0..1 "how sure"

    # sure * truth is only large where model and doctor both say pancreas.
    overlap = (sure * truth).sum()

    # Dice = 2 * overlap / (size of guess + size of truth). The + 1 avoids dividing by zero.
    dice = (2 * overlap + 1) / (sure.sum() + truth.sum() + 1)

    # Dice is a score (high is good), so 1 - dice turns it into a loss (low is good).
    return pixel_loss(guess, truth) + (1 - dice)

# 3. The model, and the optimizer that nudges its weights.
# The seed gives the same random starting weights in every run, so runs can be compared fairly.
torch.manual_seed(0)
model = UNet(width=WIDTH, slices=SLICES).to(device)

# Adam is the optimizer. lr is the learning rate: how big each nudge is.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Model:", NAME, " width:", WIDTH, " slices:", SLICES, " weights:", sum(p.numel() for p in model.parameters()), flush=True)

# Open the log file and write its header row. best_dice remembers the best exam score so far.
os.makedirs("models", exist_ok=True)
log = open("outputs/train_log_" + NAME + ".csv", "w")
log.write("epoch,train_loss,val_dice,minutes\n")
best_dice = 0

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    # 4. Training: the same four steps as learn_demo.py, on real slices.
    # The slices are put in a new random order every epoch, then handled 16 at a time.
    model.train()
    order = torch.randperm(len(train_x))
    batches = len(order) // BATCH
    loss_sum = 0
    for n in range(batches):
        # Pick the next 16 slice numbers from the shuffled order and get those slices ready.
        pick = order[n * BATCH:(n + 1) * BATCH]
        x, y = to_model(train_x[pick], train_y[pick])

        guess = model(x)             # 1. the model guesses a mask
        loss = loss_fn(guess, y)     # 2. measure how wrong it is
        optimizer.zero_grad()        #    clear the directions from the previous batch
        loss.backward()              # 3. work out which way to move each weight
        optimizer.step()             # 4. nudge every weight a little

        # Add up the loss, and print a progress line every 100 batches.
        loss_sum += loss.item()
        if n % 100 == 0:
            print("  epoch", epoch, " batch", n, "of", batches, " loss", round(loss.item(), 3), flush=True)

    # 5. Exam time: score on validation patients the model never trained on.
    # model.eval() and torch.no_grad() mean "only answer, do not learn".
    model.eval()
    overlap = guessed = true = 0
    with torch.no_grad():
        for i in range(0, len(val_x), BATCH):
            x, y = to_model(val_x[i:i + BATCH], val_y[i:i + BATCH])

            # The decision: a pixel counts as pancreas when the model is more than 0.5 sure.
            mask = (torch.sigmoid(model(x)) > 0.5).float()
            overlap += (mask * y).sum().item()
            guessed += mask.sum().item()
            true += y.sum().item()

    # One Dice score over all validation slices together.
    val_dice = 2 * overlap / (guessed + true)

    # Print the summary line for this epoch and write the same numbers to the log file.
    minutes = (time.time() - start) / 60
    train_loss = loss_sum / batches
    print("EPOCH", epoch, " train loss", round(train_loss, 3), " val Dice", round(val_dice, 3), " minutes", round(minutes, 1), flush=True)
    log.write(str(epoch) + "," + str(round(train_loss, 4)) + "," + str(round(val_dice, 4)) + "," + str(round(minutes, 1)) + "\n")
    log.flush()

    # 6. Keep the best model so far.
    # A later epoch can score worse than an earlier one, so we only save when the score improves.
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "models/" + NAME + ".pt")
        print("  saved new best model", flush=True)

log.close()
print("Finished. Best val Dice:", round(best_dice, 3))

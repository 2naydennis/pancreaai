import os
import time
import numpy as np
import torch
from unet import UNet

EPOCHS = 5
BATCH = 16

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Training on:", device, flush=True)

# 1. Load the slices you built with make_slices.py
train = np.load("data/slices/train.npz")
val = np.load("data/slices/val.npz")
train_x, train_y = torch.from_numpy(train["images"]), torch.from_numpy(train["masks"])
val_x, val_y = torch.from_numpy(val["images"]), torch.from_numpy(val["masks"])
print("Train slices:", len(train_x), " Val slices:", len(val_x), flush=True)

# Turn stored slices (0..255) back into what the model expects (0..1)
def to_model(images, masks):
    x = images.unsqueeze(1).float().div(255).to(device)
    y = masks.unsqueeze(1).float().to(device)
    return x, y

# 2. The loss: how wrong is the model's mask? Lower is better.
pixel_loss = torch.nn.BCEWithLogitsLoss()

def loss_fn(guess, truth):
    sure = torch.sigmoid(guess)                 # turn raw numbers into 0..1 "how sure"
    overlap = (sure * truth).sum()
    dice = (2 * overlap + 1) / (sure.sum() + truth.sum() + 1)
    return pixel_loss(guess, truth) + (1 - dice)

# 3. The model, and the optimizer that nudges its weights
torch.manual_seed(0)
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

os.makedirs("models", exist_ok=True)
log = open("outputs/train_log.csv", "w")
log.write("epoch,train_loss,val_dice,minutes\n")
best_dice = 0

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    # 4. Training: the same four steps as learn_demo.py, on real slices
    model.train()
    order = torch.randperm(len(train_x))
    batches = len(order) // BATCH
    loss_sum = 0
    for n in range(batches):
        pick = order[n * BATCH:(n + 1) * BATCH]
        x, y = to_model(train_x[pick], train_y[pick])

        guess = model(x)             # 1. the model guesses a mask
        loss = loss_fn(guess, y)     # 2. measure how wrong it is
        optimizer.zero_grad()
        loss.backward()              # 3. work out which way to move each weight
        optimizer.step()             # 4. nudge all 483,153 weights

        loss_sum += loss.item()
        if n % 100 == 0:
            print("  epoch", epoch, " batch", n, "of", batches, " loss", round(loss.item(), 3), flush=True)

    # 5. Exam time: score on validation patients the model never trained on
    model.eval()
    overlap = guessed = true = 0
    with torch.no_grad():
        for i in range(0, len(val_x), BATCH):
            x, y = to_model(val_x[i:i + BATCH], val_y[i:i + BATCH])
            mask = (torch.sigmoid(model(x)) > 0.5).float()
            overlap += (mask * y).sum().item()
            guessed += mask.sum().item()
            true += y.sum().item()
    val_dice = 2 * overlap / (guessed + true)

    minutes = (time.time() - start) / 60
    train_loss = loss_sum / batches
    print("EPOCH", epoch, " train loss", round(train_loss, 3), " val Dice", round(val_dice, 3), " minutes", round(minutes, 1), flush=True)
    log.write(str(epoch) + "," + str(round(train_loss, 4)) + "," + str(round(val_dice, 4)) + "," + str(round(minutes, 1)) + "\n")
    log.flush()

    # 6. Keep the best model so far
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "models/unet_best.pt")
        print("  saved new best model", flush=True)

log.close()
print("Finished. Best val Dice:", round(best_dice, 3))

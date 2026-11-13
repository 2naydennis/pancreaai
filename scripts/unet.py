# unet.py
# The model: a 2D U-Net. This file only builds it. train.py trains it.
# Run: python scripts/unet.py   (prints sizes and weight counts)
#
# Shape of the model:
#   down1 (256) ------- shortcut ------> up1 (256) -> final
#     down2 (128) ----- shortcut ----> up2 (128)
#       down3 (64) ---- shortcut ---> up3 (64)
#                  bottom (32)
# Going down, the picture shrinks and the model works out WHAT is in it.
# Going up, the picture grows back and the model marks WHERE, pixel by pixel.

# torch.nn holds ready-made network layers. "as nn" is a short nickname for it.
import torch
import torch.nn as nn

# The basic building block: two convolutions in a row.
# A convolution slides a 3x3 window over the picture and looks for patterns such as edges.
# channels_in is how many pictures go into the block, channels_out how many come out.
def block(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        # BatchNorm keeps the numbers in a steady range, which makes training smoother.
        nn.BatchNorm2d(channels_out),
        # ReLU turns negative numbers into 0. Without it the model could only learn simple rules.
        nn.ReLU(),
        nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
    )

# A class is a blueprint. __init__ lists the parts of the model,
# and forward says in which order a picture travels through them.
class UNet(nn.Module):
    # width = how many filters the first level has. 16 = small model, 32 = big model.
    # slices = how many CT slices the model sees at once. 1 = only the slice itself,
    # 3 = the slice plus the one below and the one above.
    def __init__(self, width=16, slices=1):
        super().__init__()
        w = width

        # Left side of the U: going down.
        # Each level has twice as many filters as the one above it, so it can describe more patterns.
        self.down1 = block(slices, w)
        self.down2 = block(w, w * 2)
        self.down3 = block(w * 2, w * 4)
        self.bottom = block(w * 4, w * 8)

        # MaxPool halves the picture size: 256 -> 128 -> 64 -> 32.
        self.shrink = nn.MaxPool2d(2)

        # Right side of the U: going back up. Each "grow" layer doubles the picture size.
        # The "up" blocks take twice as many pictures in, because the grown picture
        # gets glued to the matching picture from the left side first.
        self.grow3 = nn.ConvTranspose2d(w * 8, w * 4, kernel_size=2, stride=2)
        self.up3 = block(w * 8, w * 4)
        self.grow2 = nn.ConvTranspose2d(w * 4, w * 2, kernel_size=2, stride=2)
        self.up2 = block(w * 4, w * 2)
        self.grow1 = nn.ConvTranspose2d(w * 2, w, kernel_size=2, stride=2)
        self.up1 = block(w * 2, w)

        # Last layer: one number per pixel, "how sure am I this is pancreas?"
        # A high number means pancreas, a low one means not pancreas.
        self.final = nn.Conv2d(w, 1, kernel_size=1)

    def forward(self, x):
        # Going down: shrink the picture, then run a block on it.
        # d1, d2 and d3 are kept, because the right side needs them again.
        d1 = self.down1(x)                    # 256x256
        d2 = self.down2(self.shrink(d1))      # 128x128
        d3 = self.down3(self.shrink(d2))      # 64x64
        b = self.bottom(self.shrink(d3))      # 32x32

        # Going up: grow the picture, then glue on the matching left-side picture (torch.cat).
        # This shortcut gives sharp edges: the grown picture knows what is there but is blurry,
        # and the left-side picture is sharp.
        u3 = self.up3(torch.cat([self.grow3(b), d3], dim=1))    # 64x64
        u2 = self.up2(torch.cat([self.grow2(u3), d2], dim=1))   # 128x128
        u1 = self.up1(torch.cat([self.grow1(u2), d1], dim=1))   # 256x256
        return self.final(u1)

# This part only runs when you start this file directly, not when train.py borrows the model.
# It builds both sizes, sends one random picture through, and prints the weight count.
if __name__ == "__main__":
    for width in [16, 32]:
        model = UNet(width=width)
        fake_slice = torch.randn(1, 1, 256, 256)
        answer = model(fake_slice)
        print("Width", width, " goes in:", tuple(fake_slice.shape), " comes out:", tuple(answer.shape),
              " weights:", sum(p.numel() for p in model.parameters()))

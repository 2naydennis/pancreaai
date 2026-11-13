import torch
import torch.nn as nn

# The basic building block: two convolutions in a row
def block(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
        nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
    )

class UNet(nn.Module):
    # width = how many filters the first level has. 16 = small model, 32 = big model.
    # Each level down has twice as many filters as the one above it.
    def __init__(self, width=16):
        super().__init__()
        w = width

        # Left side of the U: going down
        self.down1 = block(1, w)
        self.down2 = block(w, w * 2)
        self.down3 = block(w * 2, w * 4)
        self.bottom = block(w * 4, w * 8)
        self.shrink = nn.MaxPool2d(2)

        # Right side of the U: going back up
        self.grow3 = nn.ConvTranspose2d(w * 8, w * 4, kernel_size=2, stride=2)
        self.up3 = block(w * 8, w * 4)
        self.grow2 = nn.ConvTranspose2d(w * 4, w * 2, kernel_size=2, stride=2)
        self.up2 = block(w * 4, w * 2)
        self.grow1 = nn.ConvTranspose2d(w * 2, w, kernel_size=2, stride=2)
        self.up1 = block(w * 2, w)

        # Last layer: one number per pixel, "how sure am I this is pancreas?"
        self.final = nn.Conv2d(w, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)                    # 256x256
        d2 = self.down2(self.shrink(d1))      # 128x128
        d3 = self.down3(self.shrink(d2))      # 64x64
        b = self.bottom(self.shrink(d3))      # 32x32

        # Going up: grow the picture, then glue on the matching left-side picture
        u3 = self.up3(torch.cat([self.grow3(b), d3], dim=1))    # 64x64
        u2 = self.up2(torch.cat([self.grow2(u3), d2], dim=1))   # 128x128
        u1 = self.up1(torch.cat([self.grow1(u2), d1], dim=1))   # 256x256
        return self.final(u1)

# This part only runs when you start this file directly
if __name__ == "__main__":
    for width in [16, 32]:
        model = UNet(width=width)
        fake_slice = torch.randn(1, 1, 256, 256)
        answer = model(fake_slice)
        print("Width", width, " goes in:", tuple(fake_slice.shape), " comes out:", tuple(answer.shape),
              " weights:", sum(p.numel() for p in model.parameters()))

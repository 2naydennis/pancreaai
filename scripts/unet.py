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
    def __init__(self):
        super().__init__()
        # Left side of the U: going down
        self.down1 = block(1, 16)
        self.down2 = block(16, 32)
        self.down3 = block(32, 64)
        self.bottom = block(64, 128)
        self.shrink = nn.MaxPool2d(2)

        # Right side of the U: going back up
        self.grow3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3 = block(128, 64)
        self.grow2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up2 = block(64, 32)
        self.grow1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.up1 = block(32, 16)

        # Last layer: one number per pixel, "how sure am I this is pancreas?"
        self.final = nn.Conv2d(16, 1, kernel_size=1)

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
    model = UNet()
    fake_slice = torch.randn(1, 1, 256, 256)
    answer = model(fake_slice)
    print("Goes in: ", tuple(fake_slice.shape))
    print("Comes out:", tuple(answer.shape))
    print("Weights in the model:", sum(p.numel() for p in model.parameters()))

# check_setup.py: checks that the AI tools work on this Mac.
# Run: python scripts/check_setup.py

import torch    # The tool that builds and trains AI models.
import monai    # Extra tools made for medical scans.

print("PyTorch:", torch.__version__)
print("MONAI:", monai.__version__)

# Use the Mac's fast graphics chip if it has one, otherwise the normal processor.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# Make a fake CT block full of random numbers, just for testing.
x = torch.randn(1, 1, 32, 64, 64, device=device)

# Make one layer of a 3D model. It looks for small patterns.
conv = torch.nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, padding=1).to(device)
y = conv(x)    # Push the fake block through the layer.

# If the output is the same size with 4 channels, everything works.
print("Input shape: ", tuple(x.shape))
print("Output shape:", tuple(y.shape))

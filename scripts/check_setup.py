# check_setup.py
# Checks that PyTorch and MONAI work and that the Mac's GPU can run a 3D layer.
# Run: python scripts/check_setup.py

# "import" loads a library, a box of ready-made tools.
# torch (PyTorch) builds and trains neural networks. monai adds medical-image tools.
import torch
import monai

print("PyTorch:", torch.__version__)
print("MONAI:", monai.__version__)

# Choose where the maths runs. "mps" is the Mac's GPU, which is fast.
# If PyTorch cannot see it, fall back to the slower "cpu".
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# Make a fake CT block of random numbers, only to test the machine.
# Its size: 1 scan, 1 channel, 32 slices, each 64x64 pixels.
x = torch.randn(1, 1, 32, 64, 64, device=device)

# Build one 3D convolution layer, the basic brick of a 3D U-Net.
# It slides a 3x3x3 window over the block and looks for simple patterns.
conv = torch.nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, padding=1).to(device)

# Push the fake block through the layer. y is what comes out.
y = conv(x)

# The output should have the same size as the input, with 4 channels instead of 1.
print("Input shape: ", tuple(x.shape))
print("Output shape:", tuple(y.shape))

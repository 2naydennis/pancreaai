import torch
import monai

print("PyTorch:", torch.__version__)
print("MONAI:", monai.__version__)

# Use the Mac's GPU ("mps") if PyTorch can see it, otherwise the CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# A fake CT block: 1 scan, 1 channel, 32 slices, each 64x64 pixels
x = torch.randn(1, 1, 32, 64, 64, device=device)

# One 3D convolution layer: the basic building block of a 3D U-Net
conv = torch.nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, padding=1).to(device)
y = conv(x)

print("Input shape: ", tuple(x.shape))
print("Output shape:", tuple(y.shape))

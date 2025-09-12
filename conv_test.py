import torch
from torchvision.transforms import v2

conv = torch.nn.ConvTranspose2d(in_channels=17, out_channels=17, kernel_size=3, stride=1, padding=(1, 0))
test_tensor = torch.randn(4, 17, 268, 238)
output = v2.Resize((256, 256))(test_tensor) 

print(output.shape)
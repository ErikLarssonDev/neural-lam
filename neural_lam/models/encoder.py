from collections import OrderedDict

import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    Modified U-net model
    Adapted from
    https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
    to match dimensionality of weather state (268 x 238)
    """
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        # Here input is
        
        self.encoder1 = UNet._block(in_channels, in_channels, name="enc1", padding=(1,1)) # (268 x 238)
        self.encoder2 = UNet._block(in_channels, in_channels, name="enc2", padding=(1,2)) # (268, 242)
        self.encoder3 = UNet._block(in_channels, in_channels, name="enc3", padding=(1,2)) # (268, 246)
        self.encoder4 = UNet._block(in_channels, in_channels, name="enc4", padding=(0,2)) # (264 x 250)
        self.encoder5 = UNet._block(in_channels, in_channels, name="enc5", padding=(0,2)) # (260 x 254)
        self.encoder6 = UNet._block(in_channels, in_channels, name="enc6", padding=(0,1)) # (256 x 256)
        
        self.decoder1 = UNet._block(in_channels+out_channels, out_channels, name="dec1", padding=(2,0)) # (256 x 258)
        self.decoder2 = UNet._block(in_channels+out_channels, out_channels, name="dec2", padding=(2,0)) # (260 x 254)
        self.decoder3 = UNet._block(in_channels+out_channels, out_channels, name="dec3", padding=(2,0)) # (264 x 250)
        self.decoder4 = UNet._block(in_channels+out_channels, out_channels, name="dec4", padding=(1,0)) # (268 x 246)
        self.decoder5 = UNet._block(in_channels+out_channels, out_channels, name="dec5", padding=(1,0)) # (268 x 242)
        self.decoder6 = UNet._block(in_channels+out_channels, out_channels, name="dec6", padding=(1,1)) # (268 x 238)

    def encode(self, x):
        skip_connections = []
        enc1 = self.encoder1(x)
        # print(f"ENC1: {enc1.size()}")
        skip_connections.append(enc1)
        enc2 = self.encoder2(enc1)
        # print(f"ENC2: {enc2.size()}")
        skip_connections.append(enc2)
        enc3 = self.encoder3(enc2)
        # print(f"ENC3: {enc3.size()}")
        skip_connections.append(enc3)
        enc4 = self.encoder4(enc3)
        # print(f"ENC4: {enc4.size()}")
        skip_connections.append(enc4)
        enc5 = self.encoder5(enc4)
        # print(f"ENC5: {enc5.size()}")
        skip_connections.append(enc5)
        enc5 = nn.functional.pad(enc5, (1, 1, 0, 0))
        enc6 = self.encoder6(enc5)
        # print(f"ENC6: {enc6.size()}")
        skip_connections.append(enc6)
        return enc6, skip_connections[::-1]
    
    def decode(self, x, skip_connections):
        # print(f"X: {x.size()}")
        dec1 = self.decoder1(nn.functional.pad(torch.cat((x, skip_connections[0]), dim=1), (1, 1, 0, 0)))
        # print(f"DEC1: {dec1.size()}")
        dec2 = self.decoder2(torch.cat((dec1, skip_connections[1]), dim=1))
        # print(f"DEC2: {dec2.size()}")
        dec3 = self.decoder3(torch.cat((dec2, skip_connections[2]), dim=1))
        # print(f"DEC3: {dec3.size()}")
        dec4 = self.decoder4(torch.cat((dec3, skip_connections[3]), dim=1))
        # print(f"DEC4: {dec4.size()}")
        dec5 = self.decoder5(torch.cat((dec4, skip_connections[4]), dim=1))
        # print(f"DEC5: {dec5.size()}")
        dec6 = self.decoder6(torch.cat((dec5, skip_connections[5]), dim=1))
        # print(f"DEC6: {dec6.size()}")
        return dec6

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name, padding=1):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device='cpu'
    print('#### Test Model ###')
    x = torch.rand(4, 50, 268, 238).to(device)
    model = UNet(in_channels=50, out_channels=17).to(device)
    y, skip_connections = model.encode(x)
    print(y.size())
    y = model.decode(y[:, :17, :, :], skip_connections)
    print(y.size())
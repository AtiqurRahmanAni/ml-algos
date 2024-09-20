import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding=1) -> None:
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return out


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]) -> None:
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down sampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels,
                                         out_channels=feature))
            in_channels = feature

        # Up sampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature*2,
                                   out_channels=feature,
                                   kernel_size=2,
                                   stride=2)
            )
            self.ups.append(
                DoubleConv(
                    in_channels=feature*2,
                    out_channels=feature
                )
            )

        self.bottleneck = DoubleConv(
            in_channels=features[-1],
            out_channels=features[-1] * 2
        )

        self.final_conv = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x: Tensor):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections.reverse()
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            # print(x.shape, skip_connection.shape)

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # print(x.shape, skip_connection.shape)

            x = torch.cat((x, skip_connection), dim=1)
            x = self.ups[i + 1](x)

        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    model = UNET()
    inp = torch.randn(3, 3, 160, 160)
    print(model(inp).shape)

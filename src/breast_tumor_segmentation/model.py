"""U-Net model for breast tumor segmentation."""

import torch
from torch import nn


class UNet(nn.Module):
    """U-Net architecture for binary semantic segmentation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2) -> None:
        super().__init__()

        self.conv1 = self._contract_block(in_channels, 32, 7, 3)
        self.conv2 = self._contract_block(32, 64, 3, 1)
        self.conv3 = self._contract_block(64, 128, 3, 1)

        self.upconv3 = self._expand_block(128, 64, 3, 1)
        self.upconv2 = self._expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self._expand_block(32 * 2, out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], dim=1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], dim=1))

        return upconv1

    @staticmethod
    def _contract_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    @staticmethod
    def _expand_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
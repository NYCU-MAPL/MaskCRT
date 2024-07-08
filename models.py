import torch
import torch.nn as nn

class ResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1)
        )

    def forward(self, input):
        return input + super().forward(input)
        
class Refinement(nn.Module):
    """Refinement UNet"""

    def __init__(self, in_channels, num_filters, out_channels=3):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
        self.l2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )

        self.d3 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d1 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, out_channels, 3, padding=1)
        )

    def forward(self, *input):
        if len(input) == 1:
            input = input[0]
        else:
            input = torch.cat(input, dim=1)
            
        conv1 = self.l1(input)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(deconv3 + conv2)
        deconv1 = self.d1(deconv2 + conv1)

        return deconv1
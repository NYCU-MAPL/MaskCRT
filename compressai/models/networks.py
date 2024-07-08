import torch
import torch.nn as nn
from compressai.models.utils import conv, deconv

####### For Conditioning Signal Generation in the Inter-frame Codec #######
class DownsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__(
            conv(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(0.1, inplace=True)
        )

class TopDown_extractor(nn.Module):
    def __init__(self, dims, kernel_size, scale_list):
        super().__init__()
        assert isinstance(dims, list) and isinstance(kernel_size, list) and isinstance(scale_list, list)

        self.depth = len(kernel_size)
        for i in range(self.depth):
            self.add_module('down'+str(i), DownsampleBlock(dims[i], dims[i+1], kernel_size[i], scale_list[i]))

    def forward(self, input):
        features = []
        features_size = []

        for i in range(self.depth):
            input = self._modules['down'+str(i)](input)
            features.append(input)
            features_size.append(input.shape[2:4])

        return features, features_size

class CustomTopDown_extractor(nn.Module):
    def __init__(self, input_dim=[3, 3, 3], input_down=[2, 2, 1], 
                 dims=[96, 128, 128], kernel_size=[3, 3, 3], scale_list=[1, 4, 2]):
        '''
        CustomTopDown_extractor can support multiple inputs with different resolution.
        input_dim   (list): the number of input channel for each input.
        input_down  (list): downsample ratio, which is used to align the resolution of all inputs.
        dims        (list): channel dimension for each layer.
        kernel_size (list): kernel size for each layer.
        scale_list  (list): downsample ratio for each layer.
        '''
        super().__init__()
        assert isinstance(input_dim, list)  and isinstance(input_down, list) and \
               isinstance(dims, list) and isinstance(kernel_size, list) and isinstance(scale_list, list)
        assert dims[0] % len(input_dim) == 0

        self.num_input = len(input_dim)
        for i, (dim, down) in enumerate(zip(input_dim, input_down)):
            self.add_module('conv'+str(i), DownsampleBlock(dim, dims[0] // len(input_dim), 5, down))

        self.depth = len(dims)
        dims = [dims[0]] + dims
        for i in range(self.depth):
            self.add_module('down'+str(i), DownsampleBlock(dims[i], dims[i+1], kernel_size[i], scale_list[i]))
    
    def forward(self, inputs):
        assert isinstance(inputs, list) and len(inputs) == self.num_input

        features = []
        feature_size = []

        input = []
        for i in range(self.num_input):
            input.append(self._modules['conv'+str(i)](inputs[i]))

        input = torch.cat(input, dim=1)

        for i in range(self.depth):
            input = self._modules['down'+str(i)](input)

            features.append(input)
            feature_size.append(input.shape[2:4])

        return features, feature_size

####### Mask Generator #######   
class MaskGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        self.down_1 = nn.Sequential(
            conv(in_channels, 16, 3, 1), 
            nn.ReLU(), 
            conv(16, 32, 3, 2), 
            nn.ReLU()
        )

        self.down_2 = nn.Sequential(
            conv(32, 32, 3, 1), 
            nn.ReLU(),
            conv(32, 64, 3, 2),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            conv(64, 64, 3, 1), 
            nn.ReLU(), 
            conv(64, 64, 3, 1), 
            nn.ReLU()
        )

        self.up_2 = nn.Sequential(
            deconv(128, 64, 3, 2), 
            nn.ReLU(), 
            conv(64, 32, 3, 1), 
            nn.ReLU()
        )

        self.up_1 = nn.Sequential(
            deconv(64, 32, 3, 2), 
            nn.ReLU(), 
            conv(32, 16, 3, 1), 
            nn.ReLU(),
            conv(16, out_channels, 3, 1), 
            nn.ReLU()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        down_map_1 = self.down_1(x)
        down_map_2 = self.down_2(down_map_1)

        feature = self.conv(down_map_2)

        up_map_2 = self.up_2(torch.cat([feature, down_map_2], dim=1))
        up_map_1 = self.up_1(torch.cat([up_map_2, down_map_1], dim=1))

        return self.sigmoid(up_map_1)


####### For FeatMCNet #######
def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def DownBlock(inc, outc, inplace=False):
	return torch.nn.Sequential(
        nn.Conv2d(inc, outc, 3, stride=2, padding=1),
        ResBlock(outc, inplace=inplace)
	)
 
def UpsampleSBlock(in_channels, out_channels, stride=1):
	return torch.nn.Sequential(
		nn.PReLU(),
		nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.Upsample(scale_factor=2, mode='bilinear')
	)

class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, end_with_relu=False,
                 bottleneck=False, inplace=False):
        super().__init__()
        in_channel = channel // 2 if bottleneck else channel
        self.first_layer = nn.LeakyReLU(negative_slope=slope, inplace=False)
        self.conv1 = nn.Conv2d(channel, in_channel, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.conv2 = nn.Conv2d(in_channel, channel, 3, padding=1)
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return identity + out
    
class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, leaky_relu_slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = conv1x1(in_ch, out_ch)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out
    
class FeatureExtractor(nn.Module):
    def __init__(self, incs, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(incs[0], incs[1], 3, stride=1, padding=1)
        self.res_block1 = ResBlock(incs[1], inplace=inplace)
        self.conv2 = nn.Conv2d(incs[1], incs[2], 3, stride=2, padding=1)
        self.res_block2 = ResBlock(incs[2], inplace=inplace)
        self.conv3 = nn.Conv2d(incs[2], incs[3], 3, stride=2, padding=1)
        self.res_block3 = ResBlock(incs[3], inplace=inplace)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return [layer1, layer2, layer3]

class FusionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, inplace=False):
        super(FusionUNet, self).__init__()
        
        self.bimix0 = ResidualBlock(in_channels[0], out_channels[0], 0.1, inplace)
        self.bimix1 = ResidualBlock(in_channels[1], out_channels[1], 0.1, inplace)
        self.bimix2 = ResidualBlock(in_channels[2], out_channels[2], 0.1, inplace)
        
        self.out = nn.Conv2d(out_channels[0], 3, 3, 1, 1)

        self.down1 = DownBlock(out_channels[0], out_channels[1], inplace)
        self.down2 = DownBlock(out_channels[1], out_channels[2], inplace)  
        
        self.fuse1 = UpsampleSBlock(out_channels[1], out_channels[0])
        self.fuse2 = UpsampleSBlock(out_channels[2], out_channels[1])


    def forward(self, in1, in2):
        feat0 = self.bimix0(torch.cat([in1[0], in2[0]], dim=1))
        feat1 = self.bimix1(torch.cat([in1[1], in2[1]], dim=1))
        feat2 = self.bimix2(torch.cat([in1[2], in2[2]], dim=1))

        feat1 += self.down1(feat0)
        feat2 += self.down2(feat1)
        feat1 = feat1 + self.fuse2(feat2)
        feat0 = feat0 + self.fuse1(feat1)

        frame = self.out(feat0)
        
        return frame
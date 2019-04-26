import torch
import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D
from inferno.extensions.layers.sampling import AnisotropicPool
from .base import Xcoder


def get_pooler(scale_factor):
    assert isinstance(scale_factor, (int, list, tuple))
    if isinstance(scale_factor, (list, tuple)):
        assert len(scale_factor) == 3
        assert scale_factor[0] == 1
        pooler = AnisotropicPool(downscale_factor=scale_factor[1])
    else:
        if scale_factor > 0:
            pooler = nn.MaxPool3d(kernel_size=1 + scale_factor,
                                   stride=scale_factor,
                                   padding=1)
        else:
            pooler = None
    return pooler


class Encoder(Xcoder):
    def __init__(self, in_channels, out_channels, kernel_size,
                 conv_type=ConvELU3D, scale_factor=2):
        super(Encoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      pre_conv=get_pooler(scale_factor))


class Output(Conv3D):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Output, self).__init__(in_channels, out_channels, kernel_size)

CONV_TYPES = {'vanilla': ConvELU3D,
              'conv_bn': BNReLUConv3D}

class UNetEnc3DNl(nn.Module):
    """
    3D U-Net encoder with the number of layers and output features specified by the user.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_num_fmaps,
                 fmap_growth,
                 num_layers=5,
                 scale_factor=2,
                 glob_pool = None,
                 final_activation='auto',
                 conv_type_key='vanilla'):
        """
        Parameter:
        ----------
        in_channels (int): number of input channels
        out_channels (int): number of output channels (features)
        initial_num_fmaps (int): number of feature maps of the first layer
        fmap_growth (int): growth factor of the feature maps; the number of feature maps
        in layer k is given by initial_num_fmaps * fmap_growth**k
        num_layers (int): the number of layers in the U-Net
        scale_factor (int or list / tuple): upscale / downscale factor (default: 2)
        glob_pool: the final global pooling (None, 'avg', 'max')
        final_activation:  final activation used (default: 'auto')
        conv_type_key: convolution type used (default: 'vanilla')
        """
        super(UNetEnc3DNl, self).__init__()
        assert conv_type_key in CONV_TYPES, conv_type_key
        conv_type = CONV_TYPES[conv_type_key]
        assert isinstance(scale_factor, (int, list, tuple))
        scale_factors = [scale_factor] * (num_layers-1) if isinstance(scale_factor, int) else scale_factor
        assert len(scale_factors) == num_layers -1
        scale_factors = [0] + scale_factors if isinstance(scale_factors, list) else (0,) + scale_factors
        assert all(isinstance(sfactor, (int, list, tuple)) for sfactor in scale_factors)

        # Encoders
        fe = [in_channels]
        for n in range(num_layers):
            fe.append(initial_num_fmaps * fmap_growth**n)
        encoders = []
        for n in range(num_layers):
            encoders.append(Encoder(fe[n], fe[n+1], 3, conv_type=conv_type, scale_factor=scale_factors[n]))
        self.encoders = nn.ModuleList(encoders)

        # 1*1 conv to get the needed number of channels
        self.final_conv = Conv3D(fe[num_layers], out_channels, 1)

        # Final global pooling to convert the feature maps to a feature vector
        # of the same size for any input size
        if glob_pool == 'avg':
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        elif glob_pool == 'max':
            self.global_pool = nn.AdaptiveMaxPool3d(1)
        else:
            self.global_pool = None

        # Final activation
        if final_activation == 'auto':
            self.final_activation == nn.Softmax2d()
        elif isinstance(final_activation, str):
            self.final_activation = getattr(nn, final_activation)()
        elif isinstance(final_activation, nn.Module):
            self.final_activation = final_activation
        elif final_activation is None:
            self.final_activation = None
        else:
            raise NotImplementedError

    def encode(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        x = self.final_conv(x)
        if self.global_pool is not None:
            x = self.global_pool(x)
            x = x.view(x.shape[:-2])  # remove all the singleton dims, but the last (for loss)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

    # Input is should be of shape (BATCH, 3, OTHER_DIMS) ,
    # where 3 is anchor, positive and negative samples
    def forward(self, input_):
        features_batch = []
        for triplet in input_:
            features = [self.encode(sample.unsqueeze(0))
                        for sample in triplet]
            features_batch.append(torch.cat(features))
        return (torch.stack(features_batch))


import torch
import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D
from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample, Upsample, GlobalMaskedAvgPool3d
from .base import Xcoder

CONV_TYPES = {'vanilla': ConvELU3D,
              'conv_bn': BNReLUConv3D}


def get_pooler(scale_factor):
    assert isinstance(scale_factor, (int, list, tuple))
    if isinstance(scale_factor, (list, tuple)):
        assert len(scale_factor) == 3
        assert scale_factor[0] == 1
        # we need to make sure that the scale factor conforms with the single value
        # that AnisotropicPool expects
        pooler = AnisotropicPool(downscale_factor=scale_factor[1])
    else:
        if scale_factor > 0:
            pooler = nn.MaxPool3d(kernel_size=1 + scale_factor,
                                  stride=scale_factor,
                                  padding=1)
        else:
            pooler = None
    return pooler


def get_sampler(scale_factor):
    assert isinstance(scale_factor, (int, list, tuple))
    if isinstance(scale_factor, (list, tuple)):
        assert len(scale_factor) == 3
        # we need to make sure that the scale factor conforms with the single value
        # that AnisotropicPool expects
        assert scale_factor[0] == 1
        sampler = AnisotropicUpsample(scale_factor=scale_factor[1])
    else:
        if scale_factor > 0:
            sampler = Upsample(scale_factor=scale_factor)
        else:
            sampler = None
    return sampler


class Encoder(Xcoder):
    def __init__(self, in_channels, out_channels, kernel_size,
                 conv_type=ConvELU3D, scale_factor=2):
        super(Encoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      pre_conv=get_pooler(scale_factor))


class Decoder(Xcoder):
    def __init__(self, in_channels, out_channels, kernel_size,
                 conv_type=ConvELU3D, scale_factor=2):
        super(Decoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      post_conv=get_sampler(scale_factor))


class UNet3DNlNoSkip(nn.Module):
    """
    3D U-Net architecture without skip connections
    with the number of layers specified by the user.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_num_fmaps,
                 fmap_growth,
                 num_layers=5,
                 num_decode=None,
                 scale_factor=2,
                 emb_size=None,
                 emb_fc=None,
                 glob_pool='avg',
                 final_activation='auto',
                 conv_type_key='vanilla'):
        """
        Parameter:
        ----------
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        initial_num_fmaps (int): number of feature maps of the first layer
        fmap_growth (int): growth factor of the feature maps; the number of feature maps
        in layer k is given by initial_num_fmaps * fmap_growth**k
        num_layers (int): the number of layers (excluding the base) in the U-Net
        scale_factor (int or list / tuple): upscale / downscale factor (default: 2)
        glob_pool: the final global pooling (None, 'avg', 'max')
        final_activation:  final activation used (default: 'auto')
        conv_type_key: convolution type used (default: 'vanilla')
        """
        super(UNet3DNlNoSkip, self).__init__()

        assert conv_type_key in CONV_TYPES, conv_type_key
        conv_type = CONV_TYPES[conv_type_key]
        assert isinstance(scale_factor, (int, list, tuple))
        self.scale_factor = [scale_factor] * num_layers \
                        if isinstance(scale_factor, int) else scale_factor
        assert len(self.scale_factor) == num_layers
        self.scale_factor = [0] + self.scale_factor \
                        if isinstance(self.scale_factor, list) else (0,) + self.scale_factor
        # the entry can be a tuple/list for anisotropic sampling
        assert all(isinstance(sfactor, (int, list, tuple)) for sfactor in self.scale_factor)

        # The global pooling applied on the bottleneck embedding space
        # to convert the feature maps to a feature vector
        # of the same size for any input size
        if glob_pool == 'avg':
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        elif glob_pool == 'max':
            self.global_pool = nn.AdaptiveMaxPool3d(1)
        elif glob_pool == 'avg_mask':
            self.global_pool = GlobalMaskedAvgPool3d()
        else:
            self.global_pool = None
        self.mask_value = 0

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build encoders with proper number of feature maps
        # number of feature maps for the encoders

        fe = [in_channels]
        for n in range(num_layers):
            fe.append(initial_num_fmaps * fmap_growth**n)
        encoders = []
        for n in range(num_layers):
            encoders.append(Encoder(fe[n], fe[n+1], 3, conv_type=conv_type,
                                    scale_factor=self.scale_factor[n]))
        self.encoders = nn.ModuleList(encoders)

        # Build base
        # number of base output feature maps
        f0b = initial_num_fmaps * fmap_growth**num_layers

        self.base = Encoder(fe[num_layers], f0b, 3, conv_type=conv_type,
                            scale_factor=self.scale_factor[num_layers])
        self.base_bottleneck = ConvELU3D(f0b, emb_size, 1) if emb_size is not None else None
        self.base_upsample = get_sampler(self.scale_factor[num_layers])

        emb_size = f0b if emb_size is None else emb_size
        self.emb_fc = nn.Linear(emb_size, emb_fc) if emb_fc else None

        # Decoders list
        fd = [f0b] if emb_size is None else [emb_size]
        if num_decode is None:
            num_decode = num_layers
        for n in reversed(range(num_decode)):
            fd.append(initial_num_fmaps * fmap_growth**n)
        decoders = []
        for n in range(num_decode):
            decoders.append(Decoder(fd[n], fd[n+1], 3, conv_type=conv_type,
                                    scale_factor=self.scale_factor[-n-2]))
        self.decoders = nn.ModuleList(decoders)

        # Build output
        self.output = Conv3D(fd[num_decode], out_channels, 3)
        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax3d()
        if isinstance(final_activation, str):
            self.final_activation = getattr(nn, final_activation)()
        elif isinstance(final_activation, nn.Module):
            self.final_activation = final_activation
        elif final_activation is None:
            self.final_activation = None
        else:
            raise NotImplementedError

    def get_mask(self, inp, mask_val):
        mask = (~torch.isclose(inp, torch.ones_like(inp) * mask_val, atol=5e-03)).type(torch.float)
        # pool the mask
        for i in self.scale_factor:
            pooler = get_pooler(i)
            mask = mask if pooler is None else pooler(mask)
        return mask

    def pool(self, inp, mask=None):
        if isinstance(self.global_pool, GlobalMaskedAvgPool3d):
            assert mask is not None
            inp = self.global_pool(inp, mask)
        else:
            inp = self.global_pool(inp) if self.global_pool is not None else inp
        inp = torch.flatten(inp, 1)
        return inp

    def forward(self, x, just_encode=False):
        mask = self.get_mask(x, self.mask_value) \
               if isinstance(self.global_pool, GlobalMaskedAvgPool3d) else None
        # encode
        for encoder in self.encoders:
            x = encoder(x)
        embedding = self.base(x)

        if self.base_bottleneck is not None:
            embedding = self.base_bottleneck(embedding)

        if just_encode:
            return self.pool(embedding, mask)

        # the first decoder upsample
        x = embedding if self.base_upsample is None else self.base_upsample(embedding)
        # apply decoders
        for decoder in self.decoders:
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(x)
        # apply the last layer
        x = self.output(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        pooled_emb = self.pool(embedding, mask)
        if self.emb_fc is not None:
            pooled_emb = self.emb_fc(pooled_emb)
        return x, pooled_emb

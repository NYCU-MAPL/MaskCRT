import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

# import compressai.models.transform as ts
from compressai.models.utils import conv, deconv, update_registered_buffers

from ..entropy_models import EntropyBottleneck, GaussianConditional, CompressionModel
from ..entropy_models.context_model import __CONTEXT_TYPES__
from ..layers import (
    CRSTB, RSTB, CTM_CAB, 
    ResidualBlockWithStride_DCVCDC,
    DepthConvBlock,
    ResidualBlockUpsample_DCVCDC,
    UNet
)
from .google import (
    MeanScaleHyperprior, 
    GoogleAnalysisTransform, 
    GoogleSynthesisTransform, 
    GoogleHyperAnalysisTransform, 
    GoogleHyperSynthesisTransform
)
from .networks import TopDown_extractor
from .utils import get_padding_size, get_downsampled_shape
from util.conditional_module import conditional_warping, set_condition

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

######################################### DVC #########################################
class GoogleHyperPrior(MeanScaleHyperprior):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, 
                 num_filters=128, num_features=128, num_hyperpriors=128, 
                 downsample_8=False, quant_mode='RUN', **kwargs):
        super().__init__(num_filters, num_features, **kwargs)

        self.g_a = GoogleAnalysisTransform(in_channels, num_features, num_filters, kernel_size, downsample_8)
        self.g_s = GoogleSynthesisTransform(out_channels, num_features, num_filters, kernel_size, downsample_8)
        self.h_a = GoogleHyperAnalysisTransform(num_features, num_filters, num_hyperpriors)
        self.h_s = GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)

        self.gaussian_conditional = GaussianConditional(None, quant_mode=quant_mode)
        self.entropy_bottleneck = EntropyBottleneck(num_hyperpriors, quant_mode=quant_mode)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return x_hat, (y_likelihoods, z_likelihoods)
        
######################## Swin Transformer Based Residual Coding ########################
class SwinAnalysisTransform(nn.Module):
    def __init__(self, downscale=[2, 4, 2], input_dim=3, dims=[96, 96, 192], kernel_size=[5, 3, 3], 
                 input_resolution=(128, 128), depths=[2, 2, 4], num_heads=[8, 8, 8], window_size=[8, 8, 8], 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=None, 
                 norm_layer=nn.LayerNorm):
        super(SwinAnalysisTransform, self).__init__()
        assert isinstance(depths, list) and isinstance(num_heads, list) 

        self.downscale = downscale
        
        if drop_path is None:
            dpr = [0] * sum(depths) 
        else:
            dpr = drop_path

        dims = [input_dim] + dims
        scale = 1
        for i in range(len(depths)):
            self.add_module('conv'+str(i), conv(dims[i], dims[i+1], kernel_size=kernel_size[i], stride=downscale[i]))
            scale *= downscale[i]

            self.add_module('swin'+str(i), RSTB(dim=dims[i+1],
                                                input_resolution=(input_resolution[0] // scale, input_resolution[1] // scale),
                                                depth=depths[i],
                                                num_heads=num_heads[i],
                                                window_size=window_size[i],
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                                                norm_layer=norm_layer))

    def forward(self, x, x_size=None, visual=False, vis_item=None):
        if x_size is None:
            x_size = x.shape[2:4]

        scale = 1
        visual_info = {}
        for i in range(len(self.downscale)):
            conv = self._modules['conv'+str(i)]
            x = conv(x)

            scale *= self.downscale[i]
            swin = self._modules['swin'+str(i)]
            out_dir = swin(x, (x_size[0] // scale, x_size[1] // scale), visual=visual, vis_item=vis_item)

            x = out_dir['output']
            if visual:
                visual_info['analysis'+str(i)] = out_dir['visual_info']

        if visual:
            return {
                'output'     : x,
                'visual_info': visual_info
            }

        return {'output': x}
            
class SwinSynthesisTransform(nn.Module):
    def __init__(self, upscale=[2, 4, 2], output_dim=3, dims=[96, 96, 192], kernel_size=[3, 3, 5], 
                 input_resolution=(16, 16), depths=[2, 2, 4], num_heads=[8, 8, 8], window_size=[8, 8, 8],
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=None, 
                 norm_layer=nn.LayerNorm):
        super(SwinSynthesisTransform, self).__init__()
        assert isinstance(depths, list) and isinstance(num_heads, list) 
        self.upscale = upscale
        
        if drop_path is None:
            dpr = [0] * sum(depths) 
        else:
            dpr = drop_path

        dims = dims + [output_dim]
        scale = 1
        for i in range(len(depths)):
            self.add_module('swin'+str(i), RSTB(dim=dims[i],
                                                input_resolution=(input_resolution[0] * scale, input_resolution[1] * scale),
                                                depth=depths[i],
                                                num_heads=num_heads[i],
                                                window_size=window_size[i],
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                                                norm_layer=norm_layer))
            self.add_module('deconv'+str(i), deconv(dims[i], dims[i+1], kernel_size=kernel_size[i], stride=upscale[i]))
            scale *= upscale[i]

    def forward(self, x, x_size=None, visual=False, vis_item=None):
        if x_size is None:
            x_size = x.shape[2:4]

        scale = 1
        visual_info = {}
        for i in range(len(self.upscale)):
            swin = self._modules['swin'+str(i)]
            out_dir = swin(x, (x_size[0] * scale, x_size[1] * scale), visual=visual, vis_item=vis_item)

            x = out_dir['output']
            if visual:
                visual_info['synthesis'+str(i)] = out_dir['visual_info']

            deconv = self._modules['deconv'+str(i)]
            x = deconv(x)
            scale *= self.upscale[i]

        if visual:
            return {
                'output'     : x,
                'visual_info': visual_info
            }

        return {'output': x}

class SwinHyperPriorCoder(nn.Module):
    def __init__(self, input_dim=3, scale_list=[2, 2, 2, 2, 2, 2], dims=[128, 128, 128, 128, 192, 128], 
                 kernel_size = [5, 3, 3, 3, 3, 3], depths = [2, 4, 6, 2, 2, 2], 
                 num_heads = [8, 8, 8, 16, 16, 16], window_size = [8, 8, 8, 8, 4, 4], mlp_ratio = 2., 
                 qkv_bias = True, qk_scale = None, drop_rate = 0., attn_drop_rate = 0.,
                 drop_path_rate = 0.1, norm_layer = 'LayerNorm', hyper_part=4, quant_mode='RUN'):
        super(SwinHyperPriorCoder, self).__init__()
        input_resolution = (256, 256)

        if norm_layer == 'LayerNorm':
            norm_layer = nn.LayerNorm
        elif norm_layer == 'Identity':
            norm_layer = nn.Identity
        else:
            raise NotImplementedError

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        latent_dim = dims[hyper_part-1]

        self.analysis = SwinAnalysisTransform(downscale=scale_list[:hyper_part], 
                                              input_dim=input_dim, 
                                              dims=dims[:hyper_part],
                                              kernel_size=kernel_size[:hyper_part],
                                              input_resolution=input_resolution, 
                                              depths=depths[:hyper_part],
                                              num_heads=num_heads[:hyper_part],
                                              window_size=window_size[:hyper_part],
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias,
                                              qk_scale=qk_scale,
                                              drop=drop_rate,
                                              attn_drop=attn_drop_rate,
                                              drop_path=dpr[:sum(depths[:hyper_part])],
                                              norm_layer=norm_layer
                                              )
    
        self.hyper_analysis = SwinAnalysisTransform(downscale=scale_list[hyper_part:], 
                                                    input_dim=latent_dim, 
                                                    dims=dims[hyper_part:],
                                                    kernel_size=kernel_size[hyper_part:],
                                                    input_resolution=(input_resolution[0] // np.prod(scale_list[:hyper_part]),
                                                                      input_resolution[1] // np.prod(scale_list[:hyper_part])),  
                                                    depths=depths[hyper_part:],
                                                    num_heads=num_heads[hyper_part:],
                                                    window_size=window_size[hyper_part:],
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    drop=drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    drop_path=dpr[sum(depths[:hyper_part]):],
                                                    norm_layer=norm_layer
                                                    )

        self.entropy_bottleneck = EntropyBottleneck(dims[hyper_part-1])
        self.gaussian_conditional = GaussianConditional(None, quant_mode=quant_mode)

        # reverse order
        scale_list  = scale_list[::-1]
        dims        = dims[::-1]
        kernel_size = kernel_size[::-1]
        depths      = depths[::-1]
        num_heads   = num_heads[::-1]
        window_size = window_size[::-1]
        hyper_part  = len(depths) - hyper_part
        
        self.hyper_synthesis = SwinSynthesisTransform(upscale=scale_list[:hyper_part], 
                                                      output_dim=latent_dim*2, 
                                                      dims=dims[:hyper_part],
                                                      kernel_size=kernel_size[:hyper_part],
                                                      input_resolution=(input_resolution[0] // np.prod(scale_list),
                                                                        input_resolution[1] // np.prod(scale_list)),  
                                                      depths=depths[:hyper_part],
                                                      num_heads=num_heads[:hyper_part],
                                                      window_size=window_size[:hyper_part],
                                                      mlp_ratio=mlp_ratio,
                                                      qkv_bias=qkv_bias,
                                                      qk_scale=qk_scale,
                                                      drop=drop_rate,
                                                      attn_drop=attn_drop_rate,
                                                      drop_path=dpr[:sum(depths[:hyper_part])],
                                                      norm_layer=norm_layer
                                                      )

        self.synthesis = SwinSynthesisTransform(upscale=scale_list[hyper_part:], 
                                                output_dim=input_dim, 
                                                dims=dims[hyper_part:],
                                                kernel_size=kernel_size[hyper_part:],
                                                input_resolution=(input_resolution[0] // np.prod(scale_list[hyper_part:]),
                                                                  input_resolution[1] // np.prod(scale_list[hyper_part:])),  
                                                depths=depths[hyper_part:],
                                                num_heads=num_heads[hyper_part:],
                                                window_size=window_size[hyper_part:],
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=dpr[sum(depths[:hyper_part]):],
                                                norm_layer=norm_layer
                                                )

        self.apply(self._init_weights) 

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, visual=False, vis_item=None):
        x_size = (x.shape[2], x.shape[3])
        
        ana_out = self.analysis(x, x_size, visual, vis_item)
        y = ana_out['output']
        y_size = (y.shape[2], y.shape[3])

        z = self.hyper_analysis(y, y_size, visual, vis_item)
        z_size = (z.shape[2], z.shape[3])

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.hyper_synthesis(z_hat, z_size, visual, vis_item)

        scales_hat, means_hat = params.chunk(2, 1)
        
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        syn_out = self.synthesis(y_hat, y_size, visual, vis_item)
        x_hat = syn_out['output']

        if visual:
            visual_info = {}
            visual_info.update(ana_out['visual_info'])
            visual_info.update(syn_out['visual_info'])

            return {
                'output'     : x_hat,
                'likelihoods': (y_likelihoods, z_likelihoods),
                'visual_info': visual_info
            }
        
        return x_hat, (y_likelihoods, z_likelihoods), {'mean': means_hat, 'scale': scales_hat}

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

class QualcommSwinHyperPriorCoder(nn.Module):
    def __init__(self, hyper_dim=[192, 96, 192], input_dim=3, output_dim=None, predprior_input_dim=None, 
                 scale_list=[2, 4, 2], dims=[96, 96, 192], kernel_size = [5, 3, 3], depths = [2, 4, 6], 
                 num_heads = [8, 8, 8], window_size = [8, 8, 8], mlp_ratio = 2., 
                 qkv_bias = True, qk_scale = None, drop_rate = 0., attn_drop_rate = 0.,
                 drop_path_rate = 0.1, norm_layer = 'LayerNorm', quant_mode='RUN', use_temp=False):
        super().__init__()
        input_resolution = (256, 256)

        if norm_layer == 'LayerNorm':
            norm_layer = nn.LayerNorm
        elif norm_layer == 'Identity':
            norm_layer = nn.Identity
        else:
            raise NotImplementedError

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.analysis = SwinAnalysisTransform(downscale=scale_list, 
                                              input_dim=input_dim, 
                                              dims=dims,
                                              kernel_size=kernel_size,
                                              input_resolution=input_resolution, 
                                              depths=depths,
                                              num_heads=num_heads,
                                              window_size=window_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias,
                                              qk_scale=qk_scale,
                                              drop=drop_rate,
                                              attn_drop=attn_drop_rate,
                                              drop_path=dpr,
                                              norm_layer=norm_layer
                                              )
    
        self.hyper_analysis = GoogleHyperAnalysisTransform(hyper_dim[0], hyper_dim[1], hyper_dim[2])

        self.use_temp = use_temp
        if use_temp:
            self.pred_prior = GoogleAnalysisTransform(3 if predprior_input_dim is None else predprior_input_dim, dims[-1] * 2, 128, 3)
            self.PA = nn.Sequential(
                nn.Conv2d((dims[-1] * 2) * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, dims[-1] * 2, 1)
            )

        self.entropy_bottleneck = EntropyBottleneck(hyper_dim[-1], quant_mode=quant_mode)
        self.gaussian_conditional = GaussianConditional(None, quant_mode=quant_mode)

        # reverse order
        scale_list  = scale_list[::-1]
        dims        = dims[::-1]
        kernel_size = kernel_size[::-1]
        depths      = depths[::-1]
        num_heads   = num_heads[::-1]
        window_size = window_size[::-1]
        
        self.hyper_synthesis = GoogleHyperSynthesisTransform(hyper_dim[0], hyper_dim[1], hyper_dim[2])

        self.synthesis = SwinSynthesisTransform(upscale=scale_list, 
                                                output_dim=output_dim if output_dim is not None else input_dim, 
                                                dims=dims,
                                                kernel_size=kernel_size,
                                                input_resolution=(input_resolution[0] // np.prod(scale_list),
                                                                  input_resolution[1] // np.prod(scale_list)),  
                                                depths=depths,
                                                num_heads=num_heads,
                                                window_size=window_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=dpr,
                                                norm_layer=norm_layer
                                                )

        self.apply(self._init_weights) 

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, temporal_input=None, visual=False, vis_item=None):
        x_size = (x.shape[2], x.shape[3])
        
        ana_out = self.analysis(x, x_size, visual, vis_item)
        y = ana_out['output']
        y_size = (y.shape[2], y.shape[3])

        z = self.hyper_analysis(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.hyper_synthesis(z_hat)

        if self.use_temp:
            temporal_params = self.pred_prior(temporal_input)
            params = self.PA(torch.cat([params, temporal_params], dim=1))

        scales_hat, means_hat = params.chunk(2, 1)
        
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        syn_out = self.synthesis(y_hat, y_size, visual, vis_item)
        x_hat = syn_out['output']

        if visual:
            visual_info = {}
            visual_info.update(ana_out['visual_info'])
            visual_info.update(syn_out['visual_info'])

            return {
                'output'     : x_hat,
                'likelihoods': (y_likelihoods, z_likelihoods),
                'visual_info': visual_info
            }

        return x_hat, (y_likelihoods, z_likelihoods), {'mean': means_hat, 'scale': scales_hat}

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

####################### Swin Transformer Based Conditional Coding #######################
class CondSwinAnalysisTransform(nn.Module):
    def __init__(self, downscale=[4, 2, 2], input_dim=3, dims=[96, 96, 192], kernel_size=[5, 3, 3], 
                 input_resolution=(128, 128), depths=[2, 4, 2], num_heads=[8, 8, 8], window_size=[8, 8, 8], 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=None, 
                 norm_layer=nn.LayerNorm, FA=False, num_frame_type=2, Add_CTM=False, CTM_depth=2, 
                 CTM_head=8, CA_bias=False):
        super(CondSwinAnalysisTransform, self).__init__()
        assert isinstance(depths, list) and isinstance(num_heads, list) 

        self.downscale = downscale
        
        if drop_path is None:
            dpr = [0] * sum(depths) 
        else:
            dpr = drop_path

        dims = [input_dim] + dims

        scale = 1
        for i in range(len(depths)):
            self.add_module('conv'+str(i), conv(dims[i], dims[i+1], kernel_size=kernel_size[i], stride=downscale[i]))
            scale *= downscale[i]

            self.add_module('swin'+str(i), CRSTB(dim=dims[i+1],
                                                 input_resolution=(input_resolution[0] // scale, input_resolution[1] // scale),
                                                 depth=depths[i],
                                                 num_heads=num_heads[i],
                                                 window_size=window_size[i],
                                                 mlp_ratio=mlp_ratio,
                                                 qkv_bias=qkv_bias, 
                                                 qk_scale=qk_scale,
                                                 drop=drop, attn_drop=attn_drop,
                                                 drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                                                 norm_layer=norm_layer,
                                                 FA=FA,
                                                 num_frame_type=num_frame_type))
        
        self.Add_CTM = Add_CTM
        if Add_CTM:
            self.add_module('conv_for_ctm', conv(dims[-1], dims[-1], kernel_size=kernel_size[-1], stride=1))

            self.add_module('ctm', CTM_CAB(dim=dims[-1],
                                                       depth=CTM_depth,
                                                       num_heads=CTM_head,
                                                       input_resolution=(input_resolution[0] // scale, input_resolution[1] // scale),
                                                       mlp_ratio=mlp_ratio,
                                                       drop=drop,
                                                       norm_layer=norm_layer,
                                                       CA_bias=CA_bias,
                                                       FA=FA,
                                                       num_frame_type=num_frame_type))

    def forward(self, x, conds, x_size=None, frame_type=None, visual=False, vis_item=None):
        if x_size is None:
            x_size = x.shape[2:4]

        scale = 1
        features = []
        visual_info = {}
        for i in range(len(self.downscale)):
            x = self._modules['conv'+str(i)](x)
            scale *= self.downscale[i]
            
            features.append(x)
            out_dir = self._modules['swin'+str(i)](x, conds[i], (x_size[0] // scale, x_size[1] // scale), frame_type, visual, vis_item)

            x = out_dir['output']
            if visual:
                visual_info['analysis'+str(i)] = out_dir['visual_info']
        
        if self.Add_CTM:
            x = self.conv_for_ctm(x)
            out_dir = self.ctm(x, frame_type)
            x = out_dir['output']

        if visual:
            return {
                'output'     : x,
                'visual_info': visual_info
            }

        return {'output': x, 'features': features}

class CondSwinSynthesisTransform(nn.Module):
    def __init__(self, upscale=[2, 2, 4], output_dim=3, dims=[192, 96, 96], kernel_size=[3, 3, 5], 
                 input_resolution=(16, 16), depths=[2, 4, 2], num_heads=[8, 8, 8], window_size=[8, 8, 8], 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=None, 
                 norm_layer=nn.LayerNorm, FA=False, num_frame_type=2, Add_CTM=False, CTM_depth=2, 
                 CTM_head=8, CA_bias=False):
        super(CondSwinSynthesisTransform, self).__init__()
        assert isinstance(depths, list) and isinstance(num_heads, list) 

        self.upscale = upscale
        
        if drop_path is None:
            dpr = [0] * sum(depths) 
        else:
            dpr = drop_path

        dims = dims + [output_dim]
        scale = 1

        self.Add_CTM = Add_CTM
        if Add_CTM:
            self.add_module('ctm', CTM_CAB(dim=dims[0],
                                                       depth=CTM_depth,
                                                       num_heads=CTM_head,
                                                       input_resolution=(input_resolution[0] // scale, input_resolution[1] // scale),
                                                       mlp_ratio=mlp_ratio,
                                                       drop=drop,
                                                       norm_layer=norm_layer,
                                                       CA_bias=CA_bias,
                                                       FA=FA,
                                                       num_frame_type=num_frame_type))
            
            self.add_module('deconv_for_ctm', deconv(dims[0], dims[0], kernel_size=kernel_size[0], stride=1))

        for i in range(len(depths)):
            self.add_module('swin'+str(i), CRSTB(dim=dims[i],
                                                 input_resolution=(input_resolution[0] * scale, input_resolution[1] * scale),
                                                 depth=depths[i],
                                                 num_heads=num_heads[i],
                                                 window_size=window_size[i],
                                                 mlp_ratio=mlp_ratio,
                                                 qkv_bias=qkv_bias, 
                                                 qk_scale=qk_scale,
                                                 drop=drop, attn_drop=attn_drop,
                                                 drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                                                 norm_layer=norm_layer,
                                                 FA=FA,
                                                 num_frame_type=num_frame_type))

            self.add_module('deconv'+str(i), deconv(dims[i], dims[i+1], kernel_size=kernel_size[i], stride=upscale[i]))
            scale *= upscale[i]

    def forward(self, x, conds, x_size=None, frame_type=None, visual=False, vis_item=None):
        if x_size is None:
            x_size = x.shape[2:4]

        if self.Add_CTM:
            x = self.deconv_for_ctm(x)
            out_dir = self.ctm(x, frame_type)
            x = out_dir['output']
        
        scale = 1
        features = []
        visual_info = {}
        for i in range(len(self.upscale)):
            features.append(x)
            out_dir = self._modules['swin'+str(i)](x, conds[i], (x_size[0] * scale, x_size[1] * scale), frame_type, visual, vis_item)

            x = out_dir['output']
            if visual:
                visual_info['synthesis'+str(i)] = out_dir['visual_info']

            x = self._modules['deconv'+str(i)](x)
            scale *= self.upscale[i]

        if visual:
            return {
                'output': x,
                'visual_info': visual_info
            }

        return {'output': x, 'features': features[::-1]}

class QualcommCondSwinHyperPriorCoder(QualcommSwinHyperPriorCoder):
    def __init__(self, hyper_dim=[192, 192, 192], input_dim=3, output_dim=None, cond_dim=None, predprior_input_dim=None,
                 scale_list=[4, 2, 2], dims=[96, 96, 192], kernel_size = [5, 3, 3], depths = [2, 4, 2], 
                 num_heads = [8, 8, 8], window_size = [8, 8, 8], mlp_ratio = 2., qkv_bias = True, 
                 qk_scale = None, drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1, 
                 norm_layer = 'LayerNorm', quant_mode='RUN', use_temp=False, FA=False, num_frame_type=2, 
                 Add_CTM=False, CTM_depth=2, CTM_head=8, CA_bias=False):
        super(QualcommCondSwinHyperPriorCoder, self).__init__(hyper_dim, input_dim, output_dim, predprior_input_dim, 
                                                              scale_list, dims, kernel_size, depths, num_heads, window_size, 
                                                              mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, 
                                                              drop_path_rate, norm_layer, quant_mode, use_temp)
        input_resolution = (256, 256)

        if norm_layer == 'LayerNorm':
            norm_layer = nn.LayerNorm
        elif norm_layer == 'Identity':
            norm_layer = nn.Identity
        else:
            raise NotImplementedError

        feature_dims = [input_dim if cond_dim is None else cond_dim] + dims
        self.feature_extractor = TopDown_extractor(feature_dims, kernel_size, scale_list)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        self.analysis = CondSwinAnalysisTransform(downscale=scale_list, 
                                                  input_dim=input_dim, 
                                                  dims=dims,
                                                  kernel_size=kernel_size,
                                                  input_resolution=input_resolution, 
                                                  depths=depths,
                                                  num_heads=num_heads,
                                                  window_size=window_size,
                                                  mlp_ratio=mlp_ratio,
                                                  qkv_bias=qkv_bias,
                                                  qk_scale=qk_scale,
                                                  drop=drop_rate,
                                                  attn_drop=attn_drop_rate,
                                                  drop_path=dpr,
                                                  norm_layer=norm_layer,
                                                  FA=FA,
                                                  num_frame_type=num_frame_type,
                                                  Add_CTM=Add_CTM,
                                                  CTM_depth=CTM_depth,
                                                  CTM_head=CTM_head,
                                                  CA_bias=CA_bias
                                                  )

        # reverse order
        scale_list   = scale_list[::-1]
        dims         = dims[::-1]
        kernel_size  = kernel_size[::-1]
        depths       = depths[::-1]
        num_heads    = num_heads[::-1]
        window_size  = window_size[::-1]
        feature_dims = feature_dims[::-1]

        self.synthesis = CondSwinSynthesisTransform(upscale=scale_list, 
                                                    output_dim=output_dim if output_dim is not None else input_dim, 
                                                    dims=dims,
                                                    kernel_size=kernel_size,
                                                    input_resolution=(input_resolution[0] // np.prod(scale_list),
                                                                      input_resolution[1] // np.prod(scale_list)),  
                                                    depths=depths,
                                                    num_heads=num_heads,
                                                    window_size=window_size,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    drop=drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    drop_path=dpr,
                                                    norm_layer=norm_layer,
                                                    FA=FA,
                                                    num_frame_type=num_frame_type,
                                                    Add_CTM=Add_CTM,
                                                    CTM_depth=CTM_depth,
                                                    CTM_head=CTM_head,
                                                    CA_bias=CA_bias
                                                    )

        self.FA = FA
        if FA:
            conditional_warping(self.analysis, exclude_name="ctm", discrete=True, conditions=num_frame_type, ver=2)
            conditional_warping(self.hyper_analysis, discrete=True, conditions=num_frame_type, ver=2)
            conditional_warping(self.hyper_synthesis, discrete=True, conditions=num_frame_type, ver=2)
            conditional_warping(self.synthesis, exclude_name="ctm", discrete=True, conditions=num_frame_type, ver=2)

            if use_temp:
                conditional_warping(self.pred_prior, discrete=True, conditions=num_frame_type, ver=2)
                conditional_warping(self.PA, discrete=True, conditions=num_frame_type, ver=2)
    
    def set_condition(self, cond):

        set_condition(self.analysis, cond)
        set_condition(self.hyper_analysis, cond)
        set_condition(self.hyper_synthesis, cond)
        set_condition(self.synthesis, cond)

        if self.use_temp:
            set_condition(self.pred_prior, cond)
            set_condition(self.PA, cond)

    def update(self, force=False):
        self.entropy_bottleneck.update(force=force)

        scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table=scale_table, force=force)

    def forward(self, x, cond, temporal_input=None, frame_type=None, visual=False, vis_item=None):
        if self.FA:
            self.set_condition(frame_type)
            
        # collect conditioning signal
        conds, _ = self.feature_extractor(cond)

        x_size = (x.shape[2], x.shape[3])

        ana_out = self.analysis(x, conds, x_size, frame_type, visual, vis_item)
        y = ana_out['output']
        y_size = (y.shape[2], y.shape[3])

        z = self.hyper_analysis(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.hyper_synthesis(z_hat)
        
        if self.use_temp:
            temporal_params = self.pred_prior(temporal_input)
            params = self.PA(torch.cat([params, temporal_params], dim=1))

        scales_hat, means_hat = params.chunk(2, 1)
        
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        syn_out = self.synthesis(y_hat, conds[::-1], y_size, frame_type, visual, vis_item)
        x_hat = syn_out['output']
        
        if visual:
            visual_info = {}
            visual_info.update(ana_out['visual_info'])
            visual_info.update(syn_out['visual_info'])

            return {
                'output': x_hat,
                'likelihoods': (y_likelihoods, z_likelihoods),
                'visual_info': visual_info
            }

        return x_hat, (y_likelihoods, z_likelihoods), {'mean': means_hat, 'scale': scales_hat, 
                                                       'ana_features': ana_out['features'], 'syn_features': syn_out['features'],
                                                       'cond': conds}
    
class QualcommCondSwinHyperPriorCoder_Context(QualcommCondSwinHyperPriorCoder):
    def __init__(self, hyper_dim=[192, 192, 192], input_dim=3, output_dim=None, cond_dim=None, predprior_input_dim=None,
                 scale_list=[4, 2, 2], dims=[96, 96, 192], kernel_size = [5, 3, 3], depths = [2, 4, 2], 
                 num_heads = [8, 8, 8], window_size = [8, 8, 8], mlp_ratio = 2., qkv_bias = True, 
                 qk_scale = None, drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1, 
                 norm_layer = 'LayerNorm', quant_mode='RUN', use_temp=False, FA=False, num_frame_type=2, 
                 Add_CTM=False, CTM_depth=2, CTM_head=8, CA_bias=False, context_type='Checkerboard'):
        super(QualcommCondSwinHyperPriorCoder_Context, self).__init__(hyper_dim, input_dim, output_dim, cond_dim, predprior_input_dim,
                                                                      scale_list, dims, kernel_size, depths, 
                                                                      num_heads, window_size, mlp_ratio, qkv_bias, 
                                                                      qk_scale, drop_rate, attn_drop_rate, drop_path_rate, 
                                                                      norm_layer, quant_mode, use_temp, FA, num_frame_type,
                                                                      Add_CTM, CTM_depth, CTM_head, CA_bias)
        
        self.__delattr__('gaussian_conditional')
        self.context_model = __CONTEXT_TYPES__[context_type](dims[-1], quant_mode=quant_mode)

    def update(self, force=False):
        self.entropy_bottleneck.update(force=force)

        scale_table = get_scale_table()
        self.context_model.update(scale_table=scale_table, force=force)

    def forward(self, x, cond, temporal_input=None, frame_type=None, visual=False, vis_item=None):
        if self.FA:
            self.set_condition(frame_type)
            
        # collect conditioning signal
        conds, _ = self.feature_extractor(cond)

        x_size = (x.shape[2], x.shape[3])

        ana_out = self.analysis(x, conds, x_size, frame_type, visual, vis_item)
        y = ana_out['output']
        y_size = (y.shape[2], y.shape[3])

        z = self.hyper_analysis(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.hyper_synthesis(z_hat)

        if self.use_temp:
            temporal_params = self.pred_prior(temporal_input)
            params = self.PA(torch.cat([params, temporal_params], dim=1))

        y_hat, y_likelihoods, entropy_info = self.context_model(y, params)
        
        syn_out = self.synthesis(y_hat, conds[::-1], y_size, frame_type, visual, vis_item)
        x_hat = syn_out['output']

        if visual:
            visual_info = {}
            visual_info.update(ana_out['visual_info'])
            visual_info.update(syn_out['visual_info'])

            return {
                'output': x_hat,
                'likelihoods': (y_likelihoods, z_likelihoods),
                'visual_info': visual_info
            }

        return x_hat, (y_likelihoods, z_likelihoods), {'mean': entropy_info['mean'], 'scale': entropy_info['scale'], 
                                                       'ana_features': ana_out['features'], 'syn_features': syn_out['features'],
                                                       'cond': conds}
    
#################################### DCVC-DC Intra ####################################
class DCVCDC_IntraEncoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride_DCVCDC(3, 128, stride=2, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            ResidualBlockWithStride_DCVCDC(128, 192, stride=2, inplace=inplace),
            DepthConvBlock(192, 192, inplace=inplace),
            ResidualBlockWithStride_DCVCDC(192, N, stride=2, inplace=inplace),
            DepthConvBlock(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        return self.enc_2(out)

class DCVCDC_IntraDecoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock(N, N, inplace=inplace),
            ResidualBlockUpsample_DCVCDC(N, N, 2, inplace=inplace),
            DepthConvBlock(N, N, inplace=inplace),
            ResidualBlockUpsample_DCVCDC(N, 192, 2, inplace=inplace),
            DepthConvBlock(192, 192, inplace=inplace),
            ResidualBlockUpsample_DCVCDC(192, 128, 2, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock(128, 128, inplace=inplace),
            ResidualBlockUpsample_DCVCDC(128, 16, 2, inplace=inplace),
        )

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        return self.dec_2(out)

class DCVC_Intra_NoAR(CompressionModel):
    def __init__(self, N=256, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='gaussian', z_channel=N,
                         ec_thread=ec_thread, stream_part=stream_part)

        self.enc = DCVCDC_IntraEncoder(N, inplace)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample_DCVCDC(N, N, 2, inplace=inplace),
            ResidualBlockUpsample_DCVCDC(N, N, 2, inplace=inplace),
            DepthConvBlock(N, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2, inplace=inplace),
            DepthConvBlock(N * 2, N * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 3, N * 3, inplace=inplace),
            DepthConvBlock(N * 3, N * 2, inplace=inplace),
            DepthConvBlock(N * 2, N * 2, inplace=inplace),
        )

        self.dec = DCVCDC_IntraDecoder(N, inplace)
        self.refine = nn.Sequential(
            UNet(16, 16, inplace=inplace),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )

        self.q_basic_enc = nn.Parameter(torch.ones((1, 128, 1, 1)))
        self.q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.q_scale_enc_fine = None
        self.q_basic_dec = nn.Parameter(torch.ones((1, 128, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.q_scale_dec_fine = None

    def get_q_for_inference(self, q_in_ckpt, q_index):
        q_scale_enc = self.q_scale_enc[:, 0, 0, 0] if q_in_ckpt else self.q_scale_enc_fine
        curr_q_enc = self.get_curr_q(q_scale_enc, self.q_basic_enc, q_index=q_index)
        q_scale_dec = self.q_scale_dec[:, 0, 0, 0] if q_in_ckpt else self.q_scale_dec_fine
        curr_q_dec = self.get_curr_q(q_scale_dec, self.q_basic_dec, q_index=q_index)
        return curr_q_enc, curr_q_dec

    def get_curr_q(self, q_scale, q_basic, q_index=None):
        q_scale = q_scale[q_index]
        return q_basic * q_scale

    def forward(self, x, q_in_ckpt=False, q_index=None):
        curr_q_enc, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = self.quant(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        _, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec)
        x_hat = self.refine(x_hat)

        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y, likelihood_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z, likelihood_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z

        return {
            "x_hat": x_hat,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "likelihoods":{"y":likelihood_y, "z":likelihood_z}
        }
    
    def compress(self, x, q_in_ckpt, q_index):
        curr_q_enc, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = torch.round(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()

        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result
    
    def decompress(self, bit_stream, height, width, q_in_ckpt, q_index):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        _, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        self.entropy_coder.set_stream(bit_stream)
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)

        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        return {"x_hat": x_hat}
    
    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)
    
    def quant(self, x):
        return (torch.round(x) - x).detach() + x

__CODER_TYPES__ = {'GoogleHyperPrior'                       : GoogleHyperPrior,
                   'SwinHyperPriorCoder'                    : SwinHyperPriorCoder,
                   'QualcommSwinHyperPriorCoder'            : QualcommSwinHyperPriorCoder,
                   'QualcommCondSwinHyperPriorCoder'        : QualcommCondSwinHyperPriorCoder,
                   'QualcommCondSwinHyperPriorCoder_Context': QualcommCondSwinHyperPriorCoder_Context
                   }

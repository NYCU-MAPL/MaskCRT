import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models.hub import DCVC_Intra_NoAR
from compressai.models.networks import MaskGenerator, FeatureExtractor, FusionUNet
from flownets import PWCNet, SPyNet
from models import Refinement
from SDCNet import SDCNet_3M
from util.alignment import Alignment
from util.sampler import Resampler


class Iframe_Coder(nn.Module):
    def __init__(self, model_name="DCVC-DC_Intra", quality_level=6, ms_ssim=False, write_stream=False):
        super().__init__()
        assert model_name == 'DCVC-DC_Intra', f'{model_name} is an invalid i-frame coder.'

        self.model_name = model_name

        assert quality_level in [5, 4, 3, 2], f"DCVC-DC_Intra can't support quality level {quality_level}."
        self.net = DCVC_Intra_NoAR()
        
        # if ms_ssim:
        #     self.net.load_state_dict(torch.load('./models/cvpr2023_image_ssim.pth.tar', map_location='cuda'), strict=True)
        # else: 
        #     self.net.load_state_dict(torch.load('./models/cvpr2023_image_psnr.pth.tar', map_location='cuda'), strict=True)

        if write_stream:
            self.net.update(force=True)

        self.align = Alignment(64)
        self.q_index = quality_level - 2 if model_name == 'DCVC-DC_Intra' else None

    def forward(self, coding_frame):
        coding_frame = self.align.align(coding_frame)

        if self.model_name == 'DCVC-DC_Intra':
            I_info = self.net(coding_frame, True, self.q_index)
        else:
            I_info = self.net(coding_frame)

        rec_frame = I_info['x_hat']
        rec_frame = self.align.resume(rec_frame)
    
        return rec_frame, (I_info['likelihoods']['y'], I_info['likelihoods']['z'])
        
class MENet(nn.Module):
    def __init__(self, mode='SPy'):
        super().__init__()

        if mode == 'PWC':
            self.net = PWCNet(path=None, trainable=False)
        elif mode == 'SPy':
            self.net = SPyNet(path=None, trainable=False)
        else:
            raise ValueError("Invalid ME mode: {}".format(mode))

        self.align = Alignment(16)

    def forward(self, ref_frame, current_frame):
        ref_frame = self.align.align(ref_frame)
        current_frame = self.align.align(current_frame)

        flow = self.net(ref_frame, current_frame)

        flow = self.align.resume(flow)

        return flow
        
# For DVC Motion (Intra Coding)
class MotionCoder(nn.Module):
    def __init__(self, mo_coder):
        super().__init__()

        self.net = mo_coder
        self.align = Alignment(64)

    def update(self):
        self.net.update(force=True)

    def forward(self, flow):
        flow = self.align.align(flow)

        flow_hat, likelihood_m = self.net(flow)

        flow_hat = self.align.resume(flow_hat)

        return flow_hat, likelihood_m
        
class MCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Refinement(6, 64, out_channels=3)
        self.align = Alignment(4)

    def forward(self, ref_frame, warped_frame):
        ref_frame = self.align.align(ref_frame)
        warped_frame = self.align.align(warped_frame)
        
        mc_frame = self.net(ref_frame, warped_frame)

        mc_frame = self.align.resume(mc_frame)
        
        return mc_frame

class FeatMCNet(nn.Module):
    def __init__(self, hidden_dim=[48, 64, 96], out_channel=[96, 128, 192]):
        super().__init__()
        self.feature_extractor = FeatureExtractor([3] + hidden_dim)
        self.Unet = FusionUNet([2 * dim for dim in hidden_dim], out_channel)
        self.Resampler = Resampler()
        self.align = Alignment(128)
    
    def forward(self, ref_frame, flow):
        ref_frame = self.align.align(ref_frame)
        flow = self.align.align(flow)

        feat = self.feature_extractor(ref_frame)

        feats1, feats2 = [], []
        for level, f in enumerate(feat):
            s = 2**level
            flow_scaled = F.interpolate(flow, scale_factor = 1. / s, mode="bilinear", align_corners=False) * 1. / s
            feats1.append(f)
            feats2.append(self.Resampler(f, flow_scaled))

        mc_frame = self.align.resume(self.Unet(feats1, feats2))
        return mc_frame
        
# Inter codec
class CondInterCoder(nn.Module):
    def __init__(self, res_coder, align_scale=128):
        super().__init__()

        self.net = res_coder
        self.use_temp = True if 'pred_prior' in res_coder._modules.keys() else False
        self.align = Alignment(align_scale)

    def update(self):
        self.net.update(force=True)

    def forward(self, coding_frame, cond_input, temporal_input=None, frame_type=None, visual=False, vis_item=None):
        coding_frame = self.align.align(coding_frame)
        cond_input = self.align.align(cond_input)

        if self.use_temp:
            temporal_input = self.align.align(temporal_input)

        if visual:
            out_dir = self.net(coding_frame, cond_input, temporal_input, frame_type, visual, vis_item)
            rec_frame = self.align.resume(out_dir['output'])
            
            return rec_frame, out_dir['likelihoods'], out_dir['visual_info']
        else:
            rec_frame, likelihood_r, data = self.net(coding_frame, cond_input, temporal_input, frame_type)            
            rec_frame = self.align.resume(rec_frame)
            
            return rec_frame, likelihood_r, data
        

# Motion codec
class CondMotionCoder(nn.Module):
    def __init__(self, mo_coder, write_stream=False):
        super().__init__()

        self.net = mo_coder
        self.use_temp = True if 'pred_prior' in mo_coder._modules.keys() else False
        self.align = Alignment(64)
        
    def update(self):
        self.net.update(force=True)

    def forward(self, coding_flow, cond_input, temporal_input=None, frame_type=None, visual=False, vis_item=None):
        coding_flow = self.align.align(coding_flow)
        cond_input = self.align.align(cond_input)
        
        if self.use_temp:
            assert temporal_input is not None
            temporal_input = self.align.align(temporal_input)
        
        if visual:
            out_dir = self.net(coding_flow, cond_input, temporal_input, frame_type, visual, vis_item)

            rec_frame = self.align.resume(out_dir['output'])
            
            return rec_frame, out_dir['likelihoods'], out_dir['visual_info']
        else:
            rec_frame, likelihood_m, data = self.net(coding_flow, cond_input, temporal_input, frame_type)
            rec_frame = self.align.resume(rec_frame)
            
            return rec_frame, likelihood_m, data
        

class MotionExtrapolation(nn.Module):
    def __init__(self, sequence_length, use_sdc=False, kernel_size=11):
        super().__init__()

        self.net = SDCNet_3M(sequence_length, use_sdc, kernel_size)
        self.net.__delattr__('flownet')
        self.align = Alignment(64)

    def forward(self, input_frames=None, input_flows=None, auto_warp=True):
        frames = []
        for frame in input_frames:
            frames.append(self.align.align(frame))

        flows = []
        for flow in input_flows:
            flows.append(self.align.align(flow))

        pred_frame, pred_flow = self.net(frames, flows, auto_warp)
        pred_frame = self.align.resume(pred_frame) if auto_warp else None
        pred_flow = self.align.resume(pred_flow)

        return pred_frame, pred_flow
    
class MaskGeneration(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        self.net = MaskGenerator(in_channels, out_channels)
        self.align = Alignment(4)

    def forward(self, flow):
        flow = self.align.align(flow)

        mask = self.net(flow)
        mask = self.align.resume(mask)

        return mask
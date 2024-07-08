# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time

from comet_ml import Experiment, ExistingExperiment

import flowiz as fz
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.utils import make_grid

from advance_model import *
from compressai.entropy_models import EntropyBottleneck
from compressai.models import __CODER_TYPES__
from dataloader import VideoData, VideoTestData
from trainer import Trainer
from util.estimate_bpp import estimate_bpp
from util.math import lower_bound
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.stream_helper import BitStreamIO
from util.seed import seed_everything
from util.ssim import MS_SSIM
from util.vision import PlotFlow, PlotHeatMap, save_image

plot_flow = PlotFlow().cuda()
plot_bitalloc = PlotHeatMap("RB").cuda()

lmda = {1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130, 
        5: 0.0250, 6: 0.0483, 7: 0.0932, 8: 0.1800}

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

class CompressesModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()

    def named_main_parameters(self, prefix='', include_module_name=None, exclude_module_name=None):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                if include_module_name is not None:
                    assert isinstance(include_module_name, str) or isinstance(include_module_name, list), ValueError

                    if isinstance(include_module_name, str) and (include_module_name in name):
                        yield (name, param)
                    elif isinstance(include_module_name, list) and any([_n in name for _n in include_module_name]):
                        yield (name, param)
                elif exclude_module_name is not None:
                    assert isinstance(exclude_module_name, str) or isinstance(exclude_module_name, list), ValueError
                    
                    if isinstance(exclude_module_name, str) and not (exclude_module_name in name):
                        yield (name, param)
                    elif isinstance(exclude_module_name, list) and not any([_n in name for _n in exclude_module_name]):
                        yield (name, param)
                else:
                    yield (name, param)

    def include_main_parameters(self, include_module_name=None):
        for _, param in self.named_main_parameters(include_module_name=include_module_name):
            yield param

    def exclude_main_parameters(self, exclude_module_name=None):
        for _, param in self.named_main_parameters(exclude_module_name=exclude_module_name):
            yield param

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.loss())

        return torch.stack(aux_loss).mean()

class Pframe(CompressesModel):
    def __init__(self, args, mo_coder, cond_mo_coder, res_coder, train_cfg):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not args.ssim else MS_SSIM(data_range=1.).cuda()

        self.if_model = Iframe_Coder(args.if_coder, args.quality_level, args.ssim, False)
        self.if_model.eval()

        self.MENet = MENet(args.MENet)
        self.MWNet = MotionExtrapolation(sequence_length=3)
        self.Motion = MotionCoder(mo_coder)
        self.CondMotion = CondMotionCoder(cond_mo_coder)
        self.Resampler = Resampler()
        self.MCNet = MCNet()

        self.MaskGenerator = MaskGeneration(in_channels=5, out_channels=1)
        self.Residual = CondInterCoder(res_coder)

        self.train_cfg = train_cfg

        self.frame_buffer = list()
        self.flow_buffer = list()

        self.mo_temp = self.CondMotion.use_temp
        self.res_temp = self.Residual.use_temp

    def update(self):
        self.Motion.update()
        self.CondMotion.update()
        self.Residual.update()

    def freeze(self, modules):
        '''
            modules (list): contain modules that need to freeze 
        '''
        self.requires_grad_(True)
        for module in modules:
            module = module.split('.')
            _modules = self._modules
            for m in module[:-1]:
                _modules = _modules[m]._modules
            
            for param in _modules[module[-1]].parameters(): 
                self.optimizer.state_dict()[param] = {} # remove all state (step, exp_avg, exp_avg_sg)

            _modules[module[-1]].requires_grad_(False)

    def activate(self, modules):
        '''
            modules (list): contain modules that need to activate 
        '''
        for module in modules:
            module = module.split('.')
            _modules = self._modules
            for m in module[:-1]:
                _modules = _modules[m]._modules
            _modules[module[-1]].requires_grad_(True)
                
    def motion_forward(self, ref_frame, coding_frame, predict=False, RNN=True):
        flow = self.MENet(ref_frame, coding_frame)

        if predict:
            assert len(self.frame_buffer) == 3 or len(self.frame_buffer) == 2
            if len(self.frame_buffer) == 3:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[1], self.frame_buffer[2]]
            else:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[0], self.frame_buffer[1]]

            _, pred_flow = self.MWNet(frame_buffer, 
                                      self.flow_buffer if len(self.flow_buffer) == 2 else [torch.zeros_like(self.flow_buffer[0]), self.flow_buffer[0]],
                                      True)
            
            temporal_input = self.Resampler(ref_frame, pred_flow) if self.mo_temp else None
            flow_hat, likelihood_m, data = self.CondMotion(flow, pred_flow, temporal_input=temporal_input)
        else:
            flow_hat, likelihood_m = self.Motion(flow)
        
        self.flow_buffer.append(flow_hat if RNN else flow_hat.detach())
        if len(self.flow_buffer) == 3:
            self.flow_buffer.pop(0)

        warped_frame = self.Resampler(ref_frame, flow_hat)

        mc_frame = self.MCNet(ref_frame, warped_frame)
        
        m_info = {'likelihood_m': likelihood_m, 
                  'flow': flow, 'flow_hat': flow_hat,
                  'mc_frame': mc_frame, 'warped_frame': warped_frame}
        
        if predict:
            m_info.update(data)

        return mc_frame, likelihood_m, m_info

    def forward(self, ref_frame, coding_frame, predict, RNN=True):
        mc_frame, likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame, predict, RNN)
        
        mask = self.MaskGenerator(torch.cat([m_info['flow_hat'], mc_frame], dim=1))
        temporal_input = mc_frame if self.res_temp else None
        
        res = coding_frame - mask * mc_frame
        res_hat, likelihood_r, r_info = self.Residual(res, mc_frame, temporal_input=temporal_input)
        rec_frame = res_hat + mask * mc_frame
        
        likelihoods = likelihood_m + likelihood_r

        r_info.update(
            {'res'     : res,
             'res_hat' : res_hat, 
             'mask'    : mask}
        )
        
        return rec_frame, likelihoods, m_info, r_info

    @torch.no_grad()
    def test_step(self, batch):
        if self.args.ssim: 
            dist_metric = 'MS-SSIM'
        else: 
            dist_metric = 'PSNR'

        metrics_name = [dist_metric, 'Rate', 'Mo_Rate', f'MC-{dist_metric}', f'Warped-{dist_metric}', 'Mo_Ratio', 'Res_Ratio']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        log_list = []

        dataset_name, seq_name, batch, frame_id_start = batch
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        if self.args.test_crop:
            H, W = batch.shape[-2:]
            batch = transforms.CenterCrop((128 * (H // 128), 128 * (W // 128)))(batch)
            

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)
        if self.args.visual:
            os.makedirs(self.args.save_dir + f'/{seq_name}/flow_hat', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/warped_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/motion_mean', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/motion_scale', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/residual_mean', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/residual_scale', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/likelihood_r', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/likelihood_m', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/mask', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/res', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/res_hat', exist_ok=True)

        for frame_idx in range(gop_size):
            TO_VISUALIZE = self.args.visual and frame_id_start == 1 and frame_idx < 8
            SCENE_CUT = {'videoSRC20': [43, 81], 'videoSRC21': [49, 64, 90], 'videoSRC25': [18, 84], 'videoSRC26': [62], 'videoSRC27': [40], 'videoSRC29': [71]}

            coding_frame = batch[:, frame_idx]

            # I frame
            if frame_idx == 0 or (self.args.remove_scene_cut and seq_name in SCENE_CUT and frame_id_start + frame_idx in SCENE_CUT[seq_name]):
                self.frame_buffer.clear()
                self.flow_buffer.clear()

                rec_frame, likelihoods = self.if_model(coding_frame)
                rec_frame = rec_frame.clamp(0, 1)

                r_y = estimate_bpp(likelihoods[0], input=rec_frame).mean().item()
                r_z = estimate_bpp(likelihoods[1], input=rec_frame).mean().item()
                rate = r_y + r_z

                rec_frame = rec_frame.clamp(0, 1)

                if self.args.ssim: 
                    distortion = self.criterion(rec_frame, coding_frame).mean().item()
                else: 
                    mse = self.criterion(rec_frame, coding_frame).mean().item()
                    distortion = mse2psnr(mse)

                log_list.append({f'{dist_metric}': distortion, 'Rate': rate})
            
            # P frame
            else:
                if self.args.compute_macs and frame_idx != 1:
                    def dummy_inputs(shape):
                        inputs = torch.ones(shape).cuda()
                        return {
                                'ref_frame'     : inputs, 
                                'coding_frame'  : inputs, 
                                'predict'       : True
                        }

                    self.frame_buffer = [torch.zeros_like(ref_frame)] * 3
                    self.flow_buffer = [torch.zeros([1, 2, ref_frame.shape[-2], ref_frame.shape[-1]]).to(ref_frame.device)] * 2
                    
                    macs, _ = get_model_complexity_info(self, tuple(ref_frame.shape), input_constructor=dummy_inputs)
                    print(macs)
                    raise NotImplementedError

                rec_frame, likelihoods, m_info, r_info = self(ref_frame, coding_frame, 
                                                              False if (self.args.remove_scene_cut and seq_name in SCENE_CUT and frame_id_start + frame_idx - 1 in SCENE_CUT[seq_name]) else frame_idx!=1)
                rec_frame = rec_frame.clamp(0, 1)

                mc_frame = m_info['mc_frame'].clamp(0, 1)
                warped_frame = m_info['warped_frame']
                
                m_y = estimate_bpp(likelihoods[0], input=m_info['flow']).cpu().item()
                m_z = estimate_bpp(likelihoods[1], input=m_info['flow']).cpu().item()
                r_y = estimate_bpp(likelihoods[2], input=coding_frame).cpu().item()
                r_z = estimate_bpp(likelihoods[3], input=coding_frame).cpu().item()

                m_rate = m_y + m_z
                r_rate = r_y + r_z
                rate = m_rate + r_rate

                if self.args.ssim:
                    distortion = self.criterion(rec_frame, coding_frame).mean().item()
                    mc_distortion = self.criterion(mc_frame, coding_frame).mean().item()
                    warped_distortion = self.criterion(warped_frame, coding_frame).mean().item()
                else: 
                    mse = self.criterion(rec_frame, coding_frame).mean().item()
                    distortion = mse2psnr(mse)

                    mc_mse = self.criterion(mc_frame, coding_frame).mean().item()
                    mc_distortion = mse2psnr(mc_mse)

                    warped_mse = self.criterion(warped_frame, coding_frame).mean().item()
                    warped_distortion = mse2psnr(warped_mse)

                if TO_VISUALIZE:
                    flow_map = plot_flow(m_info['flow_hat'])
                    save_image(flow_map,
                               self.args.save_dir + f'/{seq_name}/flow_hat/'
                                                    f'frame_{int(frame_id_start + frame_idx)}_flow.png', nrow=1)
                    save_image(coding_frame[0], 
                               self.args.save_dir + f'/{seq_name}/gt_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], 
                               self.args.save_dir + f'/{seq_name}/mc_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(warped_frame[0], 
                               self.args.save_dir + f'/{seq_name}/warped_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_frame[0], 
                               self.args.save_dir + f'/{seq_name}/rec_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                    if frame_idx != 1:
                        motion_mean = make_grid(torch.transpose(m_info['mean'], 0, 1), nrow=8)
                        save_image(motion_mean, self.args.save_dir + f'/{seq_name}/motion_mean/frame_{int(frame_id_start + frame_idx)}.png')

                        motion_scale = make_grid(torch.transpose(m_info['scale'], 0, 1), nrow=8)
                        save_image(motion_scale, self.args.save_dir + f'/{seq_name}/motion_scale/frame_{int(frame_id_start + frame_idx)}.png')

                    residual_mean = make_grid(torch.transpose(r_info['mean'], 0, 1), nrow=8)
                    save_image(residual_mean, self.args.save_dir + f'/{seq_name}/residual_mean/frame_{int(frame_id_start + frame_idx)}.png')

                    residual_scale = make_grid(torch.transpose(r_info['scale'], 0, 1), nrow=8)
                    save_image(residual_scale, self.args.save_dir + f'/{seq_name}/residual_scale/frame_{int(frame_id_start + frame_idx)}.png')

                    cm = plt.get_cmap('hot')
                    lll = lower_bound(likelihoods[0], 1e-9).log() / -np.log(2.)
                    rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                    plt.imshow(rate_map)
                    plt.savefig(self.args.save_dir + f'/{seq_name}/likelihood_m/frame_{int(frame_id_start + frame_idx)}_y.png')
                    plt.close()

                    cm = plt.get_cmap('hot')
                    lll = lower_bound(likelihoods[1], 1e-9).log() / -np.log(2.)
                    rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                    plt.imshow(rate_map)
                    plt.savefig(self.args.save_dir + f'/{seq_name}/likelihood_m/frame_{int(frame_id_start + frame_idx)}_z.png')
                    plt.close()

                    cm = plt.get_cmap('hot')
                    lll = lower_bound(likelihoods[2], 1e-9).log() / -np.log(2.)
                    rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                    plt.imshow(rate_map)
                    plt.savefig(self.args.save_dir + f'/{seq_name}/likelihood_r/frame_{int(frame_id_start + frame_idx)}_y.png')
                    plt.close()
                    
                    cm = plt.get_cmap('hot')
                    lll = lower_bound(likelihoods[3], 1e-9).log() / -np.log(2.)
                    rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                    plt.imshow(rate_map)
                    plt.savefig(self.args.save_dir + f'/{seq_name}/likelihood_r/frame_{int(frame_id_start + frame_idx)}_z.png')
                    plt.close()

                    m_lower = np.quantile(r_info['mask'].cpu().numpy()[0].squeeze(0), 0.00005)
                    m_upper = np.quantile(r_info['mask'].cpu().numpy()[0].squeeze(0), 0.99995)
                    plt.imshow(r_info['mask'].cpu().numpy()[0].squeeze(0), cmap='Oranges_r', vmin=m_lower, vmax=m_upper)
                    plt.axis('off')
                    plt.colorbar(shrink=0.55, pad=0.01)
                    plt.savefig(self.args.save_dir + f'/{seq_name}/mask/'
                                f'frame_{int(frame_id_start + frame_idx)}.png',
                                bbox_inches='tight', pad_inches=0.01)
                    plt.close()
                    
                    save_image(r_info['res'][0], 
                               self.args.save_dir + f'/{seq_name}/res/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    
                    save_image(r_info['res_hat'][0], 
                               self.args.save_dir + f'/{seq_name}/res_hat/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                log_list.append({dist_metric: distortion, 'Rate': rate, f'MC-{dist_metric}': mc_distortion, f'Warped-{dist_metric}': warped_distortion,
                                 'my': estimate_bpp(likelihoods[0], input=ref_frame).cpu().item(), 
                                 'mz': estimate_bpp(likelihoods[1], input=ref_frame).cpu().item(),
                                 'ry': estimate_bpp(likelihoods[2], input=ref_frame).cpu().item(), 
                                 'rz': estimate_bpp(likelihoods[3], input=ref_frame).cpu().item()})

                metrics['Mo_Rate'].append(m_rate)
                metrics[f'MC-{dist_metric}'].append(mc_distortion)
                metrics[f'Warped-{dist_metric}'].append(warped_distortion)
                metrics['Mo_Ratio'].append(m_rate/rate)
                metrics['Res_Ratio'].append(r_rate/rate)

            metrics[f'{dist_metric}'].append(distortion)
            metrics['Rate'].append(rate)

            self.frame_buffer.append(rec_frame.detach())
            if len(self.frame_buffer) == 4:
                self.frame_buffer.pop(0)

            ref_frame = rec_frame

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list}
        return logs

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        metrics_name = list(outputs[0]['metrics'].keys())  # Get all metrics' names

        rd_dict = {}

        single_seq_logs = {}
        for metrics in metrics_name:
            single_seq_logs[metrics] = {}

        single_seq_logs['LOG'] = {}
        single_seq_logs['GOP'] = {}  # Will not be printed currently
        single_seq_logs['Seq_Names'] = []

        for logs in outputs:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                
                for metrics in metrics_name:
                    rd_dict[dataset_name][metrics] = []

            for metrics in logs['metrics'].keys():
                rd_dict[dataset_name][metrics].append(logs['metrics'][metrics])

            # Initialize
            if seq_name not in single_seq_logs['Seq_Names']:
                single_seq_logs['Seq_Names'].append(seq_name)
                for metrics in metrics_name:
                    single_seq_logs[metrics][seq_name] = []
                single_seq_logs['LOG'][seq_name] = []
                single_seq_logs['GOP'][seq_name] = []

            # Collect metrics logs
            for metrics in metrics_name:
                single_seq_logs[metrics][seq_name].append(logs['metrics'][metrics])
            single_seq_logs['LOG'][seq_name].extend(logs['log_list'])
            single_seq_logs['GOP'][seq_name] = len(logs['log_list'])

        os.makedirs(self.args.save_dir + f'/report', exist_ok=True)

        for seq_name, log_list in single_seq_logs['LOG'].items():
            with open(self.args.save_dir + f'/report/{seq_name}.csv', 'w', newline='') as report:
                writer = csv.writer(report, delimiter=',')
                columns = ['frame'] + list(log_list[1].keys())
                writer.writerow(columns)

                for idx in range(len(log_list)):
                    writer.writerow([f'frame_{idx + 1}'] + list(log_list[idx].values()))

        # Summary
        logs = {}
        print_log = '{:>16} '.format('Sequence_Name')
        for metrics in metrics_name:
            print_log += '{:>12}'.format(metrics)
        print_log += '\n'

        for seq_name in single_seq_logs['Seq_Names']:
            print_log += '{:>16} '.format(seq_name[:5])

            for metrics in metrics_name:
                print_log += '{:12.4f}'.format(np.mean(single_seq_logs[metrics][seq_name]))

            print_log += '\n'
        print_log += '================================================\n'
        for dataset_name, rd in rd_dict.items():
            print_log += '{:>16} '.format(dataset_name)

            for metrics in metrics_name:
                logs['test/' + dataset_name + ' ' + metrics] = np.mean(rd[metrics])
                print_log += '{:12.4f}'.format(np.mean(rd[metrics]))

            print_log += '\n'

        print(print_log)

        with open(self.args.save_dir + f'/brief_summary.txt', 'w', newline='') as report:
            report.write(print_log)

        # self.logger.log_metrics(logs)

    def setup(self, stage):
        if stage =='test':
            #self.test_dataset = VideoTestData(self.args.dataset_root, sequence=('U', 'B', 'C', 'E', 'R', 'M'), GOP=self.args.gop)
            self.test_dataset = VideoTestData(self.args.dataset_root, sequence=('B'), GOP=self.args.gop)
        else:
            raise NotImplementedError

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        return test_loader

    def parallel(self, device_ids):
        self.if_model      = CustomDataParallel(self.if_model, device_ids=device_ids)
        self.MENet         = CustomDataParallel(self.MENet, device_ids=device_ids)
        self.MWNet         = CustomDataParallel(self.MWNet, device_ids=device_ids)
        self.Motion        = CustomDataParallel(self.Motion, device_ids=device_ids)
        self.CondMotion    = CustomDataParallel(self.CondMotion, device_ids=device_ids)
        self.MCNet         = CustomDataParallel(self.MCNet, device_ids=device_ids)
        self.MaskGenerator = CustomDataParallel(self.MaskGenerator, device_ids=device_ids)
        self.Residual      = CustomDataParallel(self.Residual, device_ids=device_ids)

def parse_args(argv):
    parser = argparse.ArgumentParser()

    # training specific
    parser.add_argument('--MENet', type=str, choices=['PWC', 'SPy'], default='SPy')
    parser.add_argument('--train_conf', type=str, default=None)
    parser.add_argument('--if_coder', type=str, default='DCVC-DC_Intra')
    parser.add_argument('--motion_coder_conf', type=str, default=None)
    parser.add_argument('--cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)
    parser.add_argument("-n", '--num_workers', type=int, default=4, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("-q", '--quality_level', type=int, default=5, help="Quality level (default: %(default)s)")
    parser.add_argument('--ssim', action='store_true', help="Optimize for MS-SSIM")

    parser.add_argument('--gpus', type=int, default=1, help="Number of GPU (default: %(default)s)")
    parser.add_argument("-data", '--dataset_root', default=None, help='Root for dataset')
    parser.add_argument('--save_dir', default=None, help='Directory for saving testing result')
    parser.add_argument('--gop', default=32, type=int)
    parser.add_argument('--visual', action='store_true', help='Visualization')

    parser.add_argument('--project_name', type=str, default='MaskCRT')
    parser.add_argument('--experiment_name', type=str, default='PSNR')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-c', '--test_crop', action='store_true')

    parser.add_argument('--compute_macs', action='store_true')
    parser.add_argument('--compute_model_size', action='store_true')
    parser.add_argument('-s', '--remove_scene_cut', action='store_true')

    args = parser.parse_args(argv)

    return args

def main(argv):
    args = parse_args(argv)
    assert args.gpus <= torch.cuda.device_count(), "Can't find enough gpus in the machine."
    
    if args.save_dir is None:
        save_root = './results/'
        if args.ssim:
            args.save_dir = os.path.join(save_root, args.experiment_name + '-Q' + str(args.quality_level))
    os.makedirs(args.save_dir, exist_ok=True)

    seed_everything(888888)
    
    gpu_ids = [0]
    for i in range(1, args.gpus):
        gpu_ids.append(i)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train config
    train_cfg = None

    # Config coders
    assert not (args.motion_coder_conf is None)
    mo_coder_cfg = yaml.safe_load(open(args.motion_coder_conf, 'r'))
    assert mo_coder_cfg['model_architecture'] in __CODER_TYPES__.keys()
    mo_coder_arch = __CODER_TYPES__[mo_coder_cfg['model_architecture']]
    mo_coder = mo_coder_arch(**mo_coder_cfg['model_params'])

    assert not (args.cond_motion_coder_conf is None)
    cond_mo_coder_cfg = yaml.safe_load(open(args.cond_motion_coder_conf, 'r'))
    assert cond_mo_coder_cfg['model_architecture'] in __CODER_TYPES__.keys()
    cond_mo_coder_arch = __CODER_TYPES__[cond_mo_coder_cfg['model_architecture']]
    cond_mo_coder = cond_mo_coder_arch(**cond_mo_coder_cfg['model_params'])

    assert not (args.residual_coder_conf is None)
    res_coder_cfg = yaml.safe_load(open(args.residual_coder_conf, 'r'))
    assert res_coder_cfg['model_architecture'] in __CODER_TYPES__.keys()
    res_coder_arch = __CODER_TYPES__[res_coder_cfg['model_architecture']]
    res_coder = res_coder_arch(**res_coder_cfg['model_params'])
    
    model = Pframe(args, mo_coder, cond_mo_coder, res_coder, train_cfg).to(device)

    if args.compute_model_size:
        modules = {'ME' : model.MENet, 'Motion Extrapolation' : model.MWNet, 'Motion' : model.Motion, 'CondMotion' : model.CondMotion, 
                   'MC' : model.MCNet, 'Residual' : model.Residual, 'I' : model.if_model, 'Whole': model}
        for key in modules.keys():
            summary(modules[key])
            param_size = 0
            for param in modules[key].parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in modules[key].buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_all_mb = (param_size + buffer_size) / 1024**2
            print(f'{key} size: {size_all_mb:.3f}MB')

        raise NotImplementedError

    # load model
    if args.ssim:
        checkpoint = torch.load(os.path.join('./models/SSIM', f'Q{args.quality_level}.pth.tar'), map_location=device)
    else:
        checkpoint = torch.load(os.path.join('./models/PSNR', f'Q{args.quality_level}.pth.tar'), map_location=device)

    ckpt = {}
    for k, v in checkpoint["state_dict"].items():
        k = k.split('.')
        k.pop(1)
        if k[0] == 'criterion':
            continue
        ckpt['.'.join(k)] = v   
    model.load_state_dict(ckpt, strict=False)

    if args.gpus >= 1 and torch.cuda.device_count() >= 1:
        model.parallel(device_ids=gpu_ids)
    
    current_epoch = 1
    trainer = Trainer(args, model, train_cfg, current_epoch, device)
    trainer.test()


if __name__ == "__main__":
    main(sys.argv[1:])

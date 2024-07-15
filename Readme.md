# MaskCRT: Masked Conditional Residual Transformer for Learned Video Compression
Accpeted to TCSVT 2024

This repository contains the source code of our TCSVT paper **MaskCRT** [arXiv](https://arxiv.org/abs/2312.15829).

## Abstract
>Conditional coding has lately emerged as the mainstream approach to learned video compression. However, a recent study shows that it may perform worse than residual coding when the information bottleneck arises. Conditional residual coding was thus proposed, creating a new school of thought to improve on conditional coding. Notably, conditional residual coding relies heavily on the assumption that the residual frame has a lower entropy rate than that of the intra frame. Recognizing that this assumption is not always true due to dis-occlusion phenomena or unreliable motion estimates, we propose a masked conditional residual coding scheme. It learns a soft mask to form a hybrid of conditional coding and conditional residual coding in a pixel adaptive manner. We introduce a Transformer-based conditional autoencoder. Several strategies are investigated with regard to how to condition a Transformer-based autoencoder for inter-frame coding, a topic that is largely under-explored. Additionally, we propose a channel transform module (CTM) to decorrelate the image latents along the channel dimension, with the aim of using the simple hyperprior to approach similar compression performance to the channel-wise autoregressive model. Experimental results confirm the superiority of our masked conditional residual transformer (termed MaskCRT) to both conditional coding and conditional residual coding. On commonly used datasets, MaskCRT shows comparable BD-rate results to VTM-17.0 under the low delay P configuration in terms of PSNR-RGB and outperforms VTM-17.0 in terms of MS-SSIM-RGB. It also opens up a new research direction for advancing learned video compression.

## Install

```bash
git clone https://github.com/NYCU-MAPL/MaskCRT
conda env create -n TransCodec python=3.8.8
conda activate TransCodec
cd MaskCRT
./install.sh
```

## Pre-trained Weights
Download the following pre-trained models and put them in the corresponding folder in `./models`.
|         Metrics         | Quality 5 | Quality 4 | Quality 3 | Quality 2 |
|:-----------------------:|-----------|-----------|-----------|-----------|
|     PSNR-RGB            | [Q5](https://github.com/NYCU-MAPL/MaskCRT/releases/download/v1.0/PSNR_Q5.pth.tar) | [Q4](https://github.com/NYCU-MAPL/MaskCRT/releases/download/v1.0/PSNR_Q4.pth.tar) | [Q3](https://github.com/NYCU-MAPL/MaskCRT/releases/download/v1.0/PSNR_Q3.pth.tar) | [Q2](https://github.com/NYCU-MAPL/MaskCRT/releases/download/v1.0/PSNR_Q2.pth.tar) |
|     MS-SSIM-RGB         | [Q5](https://github.com/NYCU-MAPL/MaskCRT/releases/download/v1.0/SSIM_Q5.pth.tar) | [Q4](https://github.com/NYCU-MAPL/MaskCRT/releases/download/v1.0/SSIM_Q4.pth.tar) | [Q3](https://github.com/NYCU-MAPL/MaskCRT/releases/download/v1.0/SSIM_Q3.pth.tar) | [Q2](https://github.com/NYCU-MAPL/MaskCRT/releases/download/v1.0/SSIM_Q2.pth.tar) |


## Example Usage
Example of testing the pre-trained highest rate model:
`python CondMo_CondResTransformerCodec.py --MENet SPy --if_coder DCVC-DC_Intra --motion_coder_conf ./config/DVC_motion.yml --cond_motion_coder_conf ./config/CondMo_predprior.yml --residual_coder_conf ./config/CondInter_with_CTM_Checkerboard_Context.yml -data [dataset_root] -n 0 --quality_level 5 --gpus 1 --gop 32 --test -c`

* Organize your test dataset according to the following file structure:
```
[dataset_root]/
    |-UVG/
    |-HEVC-B/
    |-HEVC-C/
    |-HEVC-E/
    |-HEVC-RGB/
    |-MCL-JCV/
```

* Add `--ssim` to evaluate MS-SSIM model


## Citation
If you find our project useful, please cite the following paper.
```
@article{MaskCRT,
  title={MaskCRT: Masked Conditional Residual Transformer for Learned Video Compression},
  author={Chen, Yi-Hsin and Xie, Hong-Sheng and Chen, Cheng-Wei and Gao, Zong-Lin and Benjak, Martin and Peng, Wen-Hsiao and Ostermann, J{\"o}rn},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2024},
  doi={10.1109/TCSVT.2024.3427426}
  }
```

## Ackownledgement
Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the intra codecs are from [DCVC-DC](https://github.com/microsoft/DCVC/tree/main/DCVC-DC). We thank the authors for open-sourcing their code.

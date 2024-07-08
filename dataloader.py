import os
import random

from glob import glob

from PIL import Image
from torch import stack
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as torchData
from torchvision import transforms

from util.seed import seed_everything
from util.vision import imgloader, rgb_transform


class VideoData(torchData):
    """Video Dataset

    Args:
        root
        mode
        frames
        transform
    """

    def __init__(self, root, frames, transform=rgb_transform, epoch_ratio=1):
        super().__init__()
        self.folder = glob(root + 'img/*/*/')
        self.frames = frames
        self.transform = transform

        assert 0 < epoch_ratio and epoch_ratio <= 1
        self.epoch_ratio = epoch_ratio

    def __len__(self):
        return int(len(self.folder) * self.epoch_ratio)

    @property
    def info(self):
        gop = self[0]
        return "\nGop size: {}".format(gop.shape)

    def __getitem__(self, index):
        path = self.folder[index]
        seed = random.randint(0, 1e9)
        imgs = []
        for f in range(self.frames):
            seed_everything(seed)
            file = path + str(f) + '.png'
            imgs.append(self.transform(imgloader(file)))

        return stack(imgs)

class VideoTestData(torchData):
    def __init__(self, root, first_gop=False, sequence=('U', 'B'), GOP=32):
        super(VideoTestData, self).__init__()
        
        assert GOP in [12, 16, 32], ValueError
        self.root = root

        self.seq_name = []
        seq_len = []
        gop_size = []
        dataset_name_list = []
        if 'U' in sequence:
            self.seq_name.extend(['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'])
            if GOP in [12, 16]:
                seq_len.extend([600, 600, 600, 600, 600, 300, 600])
            else:
                seq_len.extend([96]*7)
        
            gop_size.extend([GOP]*7)
            dataset_name_list.extend(['UVG']*7)

        if 'B' in sequence:
            self.seq_name.extend(['Kimono1', 'BQTerrace', 'Cactus', 'BasketballDrive', 'ParkScene'])
            if GOP in [12, 16]:
                seq_len.extend([100]*5)
            else:
                seq_len.extend([96]*5)

            gop_size.extend([GOP]*5)
            dataset_name_list.extend(['HEVC-B']*5)

        if 'C' in sequence:
            self.seq_name.extend(['BasketballDrill_832x480', 'BQMall_832x480', 'PartyScene_832x480', 'RaceHorses_832x480'])
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
            else:
                seq_len.extend([96]*4)

            gop_size.extend([GOP]*4)
            dataset_name_list.extend(['HEVC-C']*4)

        if 'E' in sequence:
            self.seq_name.extend(['FourPeople_1280x720', 'Johnny_1280x720', 'KristenAndSara_1280x720'])
            if GOP in [12, 16]:
                seq_len.extend([100]*3)
            else:
                seq_len.extend([96]*3)

            gop_size.extend([GOP]*3)
            dataset_name_list.extend(['HEVC-E']*3)

        if 'R' in sequence:
            self.seq_name.extend(['DucksAndLegs', 'EBULupoCandlelight', 'EBURainFruits', 'Kimono1', 'OldTownCross', 'ParkScene'])
            if GOP in [12, 16]:
                seq_len.extend([100]*6)
            else:
                seq_len.extend([96]*6)

            gop_size.extend([GOP]*6)
            dataset_name_list.extend(['HEVC-RGB']*6)

        if 'M' in sequence:
            MCL_list = []
            for i in range(1, 31):
               MCL_list.append('videoSRC'+str(i).zfill(2))
            
            self.seq_name.extend(MCL_list)
            if GOP in [12, 16]:
               seq_len.extend([150, 150, 150, 150, 125, 125, 125, 125, 125, 150,
                               150, 150, 150, 150, 150, 150, 120, 125, 150, 125,
                               120, 120, 120, 120, 120, 150, 150, 150, 120, 150])
            else:
               seq_len.extend([96]*30)
               
            gop_size.extend([GOP]*30)
            dataset_name_list.extend(['MCL_JCV']*30)

        
        seq_len = dict(zip(self.seq_name, seq_len))
        gop_size = dict(zip(self.seq_name, gop_size))
        dataset_name_list = dict(zip(self.seq_name, dataset_name_list))

        self.gop_list = []

        for seq_name in self.seq_name:
            if first_gop:
                gop_num = 1
            else:
                gop_num = seq_len[seq_name] // gop_size[seq_name]
                
            for gop_idx in range(gop_num):
                self.gop_list.append([dataset_name_list[seq_name],
                                      seq_name,
                                      1 + gop_size[seq_name] * gop_idx,
                                      1 + gop_size[seq_name] * (gop_idx + 1)])
        
    def __len__(self):
        return len(self.gop_list)

    def __getitem__(self, idx):
        dataset_name, seq_name, frame_start, frame_end = self.gop_list[idx]
        imgs = []

        for frame_idx in range(frame_start, frame_end):
            raw_path = os.path.join(self.root, dataset_name, seq_name, 'frame_{:d}.png'.format(frame_idx))
            imgs.append(transforms.ToTensor()(imgloader(raw_path)))

        return dataset_name, seq_name, stack(imgs), frame_start
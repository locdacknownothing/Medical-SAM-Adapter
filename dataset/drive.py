import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import init_point_sampling

class DRIVE(Dataset):
    def __init__(self, args, data_path, transform = None, transform_msk = None, mode = 'Training', prompt = 'click', plane = False, point_num=3):
        self.mode = mode
        sub_folder = 'train' if mode == 'Training' else 'test'
        self.data_path = os.path.join(data_path, sub_folder)
        self.name_list = os.listdir(os.path.join(self.data_path, 'labels'))
        # Exclude checkpoints or hidden files
        self.name_list = [f for f in self.name_list if f.endswith('.png')]
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.point_num = point_num

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        msk_filename = self.name_list[index]
        name = msk_filename.replace('_manual1.png', '')
        if self.mode == 'Training':
            img_filename = name + '_training.tif'
        else:
            img_filename = name + '_test.tif'

        img_path = os.path.join(self.data_path, 'images', img_filename)
        msk_path = os.path.join(self.data_path, 'labels', msk_filename)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.prompt == 'click':
            pt, point_label = init_point_sampling(np.array(mask) / 255, get_point=self.point_num)
        else:
            pt = torch.zeros(self.point_num, 2)
            point_label = torch.zeros(self.point_num, dtype=torch.int)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask).int()

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }

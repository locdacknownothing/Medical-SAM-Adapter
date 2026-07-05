import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click, init_point_sampling


# def pad_to_square(image):
#     w, h = image.size
#     if w == h:
#         return image
#     max_side = max(w, h)
#     if image.mode == 'RGB':
#         new_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
#     else:
#         new_img = Image.new(image.mode, (max_side, max_side), 0)
#     offset = ((max_side - w) // 2, (max_side - h) // 2)
#     new_img.paste(image, offset)
#     return new_img


class STARE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        self.data_path = data_path
        self.name_list = os.listdir(os.path.join(data_path,'masks'))
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index].split('.')[0]

        img_path = os.path.join(self.data_path, 'images',name+'.ppm')
        
        msk_path = os.path.join(self.data_path, 'masks', name+'.ah.pgm')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        # img = pad_to_square(img)
        # mask = pad_to_square(mask)
        # newsize = (self.img_size, self.img_size)
        # mask = mask.resize(newsize, Image.NEAREST)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }


class STARE_AUG(Dataset):
    def __init__(self, args, data_path, transform = None, transform_msk = None, mode = 'Training', prompt = 'click', plane = False, point_num=3):
        sub_folder = 'train' if mode == 'Training' else 'test'
        self.data_path = os.path.join(data_path, sub_folder)
        self.name_list = os.listdir(os.path.join(self.data_path, 'labels'))
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.point_num = point_num

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        msk_filename = self.name_list[index]
        name = msk_filename.replace('.ah.png', '')
        img_filename = name + '.png'

        img_path = os.path.join(self.data_path, 'images', img_filename)
        msk_path = os.path.join(self.data_path, 'labels', msk_filename)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # img = pad_to_square(img)
        # mask = pad_to_square(mask)
        # newsize = (self.img_size, self.img_size)
        # mask = mask.resize(newsize, Image.NEAREST)

        if self.prompt == 'click':
            pt, point_label = init_point_sampling(np.array(mask) / 255, get_point=self.point_num)

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
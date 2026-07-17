import os
import sys
import torch
from monai.data import CacheDataset, ThreadDataLoader, load_decathlon_datalist, set_track_meta
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from utils import device

def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if net == 'sam':
        from models.sam import SamPredictor, sam_model_registry
        from models.sam.utils.transforms import ResizeLongestSide
        options = ['default','vit_b','vit_l','vit_h']
        if args.encoder not in options:
            raise ValueError("Invalid encoder option. Please choose from: {}".format(options))
        else:
            net = sam_model_registry[args.encoder](args,checkpoint=args.sam_ckpt).to(device)

    elif net == 'efficient_sam':
        from models.efficient_sam import sam_model_registry
        options = ['default','vit_s','vit_t']
        if args.encoder not in options:
            raise ValueError("Invalid encoder option. Please choose from: {}".format(options))
        else:
            net = sam_model_registry[args.encoder](args)

    elif net == 'mobile_sam':
        from models.MobileSAMv2.mobilesamv2 import sam_model_registry
        options = ['default','vit_h','vit_l','vit_b','tiny_vit','efficientvit_l2','PromptGuidedDecoder','sam_vit_h']
        if args.encoder not in options:
            raise ValueError("Invalid encoder option. Please choose from: {}".format(options))
        else:
            net = sam_model_registry[args.encoder](args,checkpoint=args.sam_ckpt)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net

def get_decath_loader(args):

    train_transforms = Compose(
        [   
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_size, args.roi_size, args.chunk),
                pos=1,
                neg=1,
                num_samples=args.num_sample,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )



    data_dir = args.data_path
    split_JSON = "dataset_0.json"

    datasets = os.path.join(data_dir, split_JSON)
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=2, cache_rate=1.0, num_workers=0
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False)

    return train_loader, val_loader, train_transforms, val_transforms, datalist, val_files

def get_siren(args):
    wrapper = get_network(args, 'siren', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device), distribution = args.distributed)
    '''load init weights'''
    checkpoint = torch.load('./logs/siren_train_init_2022_08_19_21_00_16/Model/checkpoint_best.pth')
    wrapper.load_state_dict(checkpoint['state_dict'],strict=False)
    '''end'''

    '''load prompt'''
    checkpoint = torch.load('./logs/vae_standard_refuge1_2022_08_21_17_56_49/Model/checkpoint500')
    vae = get_network(args, 'vae', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device), distribution = args.distributed)
    vae.load_state_dict(checkpoint['state_dict'],strict=False)
    '''end'''

    return wrapper, vae


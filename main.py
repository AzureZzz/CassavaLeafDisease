import os
import torch
import argparse
import logging
import numpy as np
import random

from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue, RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

from dataloader import get_dataset
from models import get_model
from trainer import Trainer


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_logging():
    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s\n%(levelname)s:%(message)s'
        format='%(levelname)s:%(message)s'
    )
    return logging


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=10)

    parser.add_argument('--task', type=str, default='task1')
    parser.add_argument('--action', type=str, default='train')

    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_val', type=int, default=32)
    parser.add_argument('--val_epoch', type=int, default=2)
    parser.add_argument('--save_model_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--tensorboard_dir', type=str, default='result/runs')
    parser.add_argument('--save_path', type=str, default='checkpoints')

    parser.add_argument('--cuda_ids', type=str, default='0')
    parser.add_argument('--DataParallel', type=bool, default=True)
    return parser.parse_args()


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    logging = get_logging()
    seed_everything(args.seed)

    transforms_train = Compose([
        RandomResizedCrop(args.img_size, args.img_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)
    transforms_val = Compose([
        CenterCrop(args.img_size, args.img_size, p=1.),
        Resize(args.img_size, args.img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

    train_loader, val_loader = get_dataset(args.data_path, args.batch_size, args.batch_size_val, transforms_train,
                                           transforms_val)
    net = get_model(True, device)

    trainer = Trainer(net, train_loader, val_loader, args, device, logging)
    trainer.train()


if __name__ == '__main__':
    main()

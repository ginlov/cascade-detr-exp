# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .dataset import SFCHD

def build_dataset(image_set, args):
    if args.dataset_name == 'SFCHD':
        return SFCHD(root=args.data_path, image_folder=args.image_path, train=(image_set == 'train'))


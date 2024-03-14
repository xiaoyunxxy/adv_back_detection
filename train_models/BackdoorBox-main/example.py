'''
This is the example code of benign training and poisoned training on torchvision.datasets.DatasetFolder.
Dataset is CIFAR-10.
Attack method is BadNets.
'''


import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import argparse

import core

from loader import dataset_loader, network_loader


dataset = torchvision.datasets.DatasetFolder



def get_args_parser():
    parser = argparse.ArgumentParser('test', add_help=False)
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--dataset', default='gtsrb', type=str, help='dataset name')
    parser.add_argument('--network', default='resnet18', type=str, help='network name')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    parser.add_argument('--data_root', default='../../../data/', type=str, help='path to dataset')
    parser.add_argument('--epoch', default=60, type=int, help='epoch number')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
    return parser

args = get_args_parser()
args = args.parse_args()

train_loader, test_loader = dataset_loader(args)


pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 1
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0

badnets = core.BadNets(
    train_dataset=train_loader,
    test_dataset=test_loader,
    model=core.models.resnet18(num_classes=43),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    # poisoned_transform_index=0,
    poisoned_target_transform_index=0,
    schedule=None,
    seed=666
)

poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()

# x, y = poisoned_train_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()

# x, y = poisoned_test_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()

# train benign model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 256,
    'num_workers': 16,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 1,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_benign_DatasetFolder-CIFAR10'
}

badnets.train(schedule)

# train attacked model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 256,
    'num_workers': 16,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 1,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisoned_DatasetFolder-CIFAR10'
}

badnets.train(schedule)
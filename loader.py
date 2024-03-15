#!/usr/bin/env python

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# Custom package
from models.vgg import vgg16
from models.resnet import resnet18, resnet50
from models.mobilenetv2 import MobileNetV2
from models.anp_batchnorm import NoisyBatchNorm2d


def network_loader(args):
    print('Pretrained', args.pretrained)
    print('Batchnorm', args.batchnorm)
    if args.network == "resnet18":
        print('ResNet18 Network')
        return resnet18(num_classes=args.num_classes, norm_layer=NoisyBatchNorm2d)
    elif args.network == "resnet50":
        print('ResNet50 Network')
        return resnet50(num_classes=args.num_classes, norm_layer=NoisyBatchNorm2d)
    elif args.network == "vgg16":
        print('VGG16 Network')
        return vgg16(num_classes=args.num_classes, norm_layer=NoisyBatchNorm2d)
    elif args.network == "mobilenet":
        print('MobileNetV2 Network')
        return MobileNetV2(num_classes=args.num_classes, norm_layer=NoisyBatchNorm2d)


def dataset_loader(args):
    # Setting Dataset Required Parameters
    transforms_list = []
    transforms_list_test = []
    if args.dataset == "cifar10":
        args.num_classes = 10
        args.img_size  = 32
        args.channel   = 3
        args.mean = [0.491, 0.482, 0.446]
        args.std = [0.247, 0.243, 0.261]
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.img_size  = 32
        args.channel   = 3
        args.mean = None
        args.std = None
        transforms_list_test.append(transforms.Resize(32))
        transforms_list_test.append(transforms.CenterCrop(32))
    elif args.dataset == "imagenet200":
        args.num_classes = 200
        args.img_size  = 224
        args.channel   = 3
        args.mean = [0.4802, 0.4481, 0.3975]
        args.std = [0.2302, 0.2265, 0.2262]
        transforms_list_test.append(transforms.Resize(256))
        transforms_list_test.append(transforms.CenterCrop(224))

    
    transforms_list.append(transforms.RandomResizedCrop(args.img_size))
    transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())        
    transforms_list_test.append(transforms.ToTensor())

    if args.mean is not None and args.std is not None:
        transforms_list.append(transforms.Normalize(args.mean, args.std))
        transforms_list_test.append(transforms.Normalize(args.mean, args.std))        

    transform_train = transforms.Compose(transforms_list)
    transform_test = transforms.Compose(transforms_list_test)


    # Full Trainloader/Testloader
    dataset_train = dataset(args, True,  transform_train)
    dataset_test = dataset(args, False, transform_test)
    

    # trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    # testloader  = torch.utils.data.DataLoader(dataset_test,  batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    return dataset_train, dataset_test


def dataset(args, train, transform):
        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)

        if args.dataset == "gtsrb":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/GTSRB/Train' if train \
                else args.data_root+'/GTSRB/val4imagefolder', transform=transform)
            # return torchvision.datasets.GTSRB(root=args.data_root+'gtsrb_torch', split='train' if train \
            #  else 'test', transform=transform, download=True)

        elif args.dataset == "imagenet200":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/imagenet200/train' if train \
                                    else args.data_root + '/imagenet200/val', transform=transform)
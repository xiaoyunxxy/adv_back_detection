import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison_cifar
import data.poison_gtsrb as poison_gtsrb

from loader import dataset_loader

from wanet_eval import eval as wa_eval
from iad_eval import eval as iad_eval
from train_models.IAD.networks.models import Generator
from train_models.IAD.dataloader import get_dataloader as iad_get_dataloader
from train_models.wanet.utils.dataloader import get_dataloader as wanet_get_dataloader

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg16'])
parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--data-root", type=str, default="../data/")
parser.add_argument("--num-classes", type=int, default=10)
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--lr', type=float, default=0.2, help='the learning rate for mask optimization')
parser.add_argument('--nb-iter', type=int, default=2000, help='the number of iterations for training')
parser.add_argument('--print-every', type=int, default=200, help='print results every few iterations')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output-dir', type=str, default='logs/models/')
parser.add_argument("--num-workers", type=float, default=10)

parser.add_argument('--trigger-info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='benign',
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')
parser.add_argument("--attack-mode", type=str, default="all2one", help="all2one or all2all")
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument('--eps', type=float, default=0.4)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.2)

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    if args.trigger_info:
        trigger_info = torch.load(args.trigger_info, map_location=device)
    else:
        if args.poison_type in ['badnets', 'blend']:
            triggers = {'badnets': 'checkerboard_1corner',
                        'clean-label': 'checkerboard_4corner',
                        'blend': 'gaussian_noise'}
            trigger_type = triggers[args.poison_type]
            pattern, mask = poison_cifar.generate_trigger(trigger_type=trigger_type)
            trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}
        else:
            trigger_info = None

    orig_train, clean_test = dataset_loader(args)
    sub_test, _ = random_split(dataset=clean_test, lengths=[args.val_frac, 1-args.val_frac], generator=torch.Generator().manual_seed(0))
    sub_train, _ = random_split(dataset=orig_train, lengths=[args.val_frac, 1-args.val_frac], generator=torch.Generator().manual_seed(0))
    clean_val = sub_train

    # poisoned test set.
    if args.dataset == 'cifar10':
        poison_test = poison_cifar.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    elif args.dataset == 'gtsrb':
        poison_test = poison_gtsrb.add_predefined_trigger_gtsrb(clean_test, trigger_info)
    elif args.dataset == 'imagenet200':
        poison_test = clean_test
    else:
        raise ValueError('Wrong dataset.')

    random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples=args.print_every * args.batch_size)
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=args.num_workers)

    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=args.num_workers)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # Step 2: load model checkpoints and trigger info
    if args.poison_type == 'wanet':
        mode = args.attack_mode
        args.ckpt_folder = os.path.join(args.checkpoint, args.dataset)
        args.ckpt_path = os.path.join(args.ckpt_folder, "{}_{}_morph.pth.tar".format(args.dataset, mode))
        state_dict = torch.load(args.ckpt_path, map_location=device)["netC"]
    elif args.poison_type == 'iad':
        path_model = os.path.join(
            args.checkpoint, args.dataset, args.attack_mode, "{}_{}_ckpt.pth.tar".format(args.attack_mode, args.dataset)
        )
        state_dict = torch.load(path_model, map_location=device)["netC"]
    else:   
        state_dict = torch.load(args.checkpoint, map_location=device)
    net = getattr(models, args.arch)(num_classes=args.num_classes, norm_layer=models.MaskNoisyBatchNorm2d)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.eps / args.steps)

    for n, v in parameters:
        if "neuron_mask" in n or "neuron_noise" in n:
            v.requires_grad = True
        else:
            v.requires_grad = False

    # Step 3: train backdoored models
    print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonACC \t CleanACC')
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    for i in range(nb_repeat):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=clean_val_loader,
                                           mask_opt=mask_optimizer, noise_opt=noise_optimizer)
        if args.poison_type == 'iad':
            cl_test_acc, po_test_acc = iad_test(args, net)
        elif args.poison_type == 'wanet':
            cl_test_acc, po_test_acc = wanet_test(args, net)
        else:
            cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        end = time.time()
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_acc, cl_test_acc))
    save_mask_scores(net.state_dict(), os.path.join(args.output_dir, 'mask_values.txt'))


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.5):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskNoisyBatchNorm2d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskNoisyBatchNorm2d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskNoisyBatchNorm2d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskNoisyBatchNorm2d):
            module.reset(rand_init=rand_init, eps=args.eps)


def mask_train(model, criterion, mask_opt, noise_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if args.eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = args.alpha * loss_nat + (1 - args.alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def wanet_test(opt, netC):
    opt.s=0.5
    opt.grid_rescale=1
    opt.target_label=0
    opt.cross_ratio=2
    opt.input_height = args.img_size
    opt.input_width = args.img_size
    opt.input_channel = args.channel

     # Dataset
    opt.bs = opt.batch_size
    test_dl = wanet_get_dataloader(opt, False)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, [30, 45], 0.1)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoint, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    
    if os.path.exists(opt.ckpt_path):
        state_dict = torch.load(opt.ckpt_path)
        # netC.load_state_dict(state_dict["netC"])
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]
    else:
        print("Pretrained model doesnt exist")
        exit()

    
    acc_clean, acc_bd = wa_eval(
        netC,
        optimizerC,
        schedulerC,
        test_dl,
        noise_grid,
        identity_grid,
        opt,
    )
    return acc_clean, acc_bd


def iad_test(opt, netC):
    opt.lr_G = 1e-2
    opt.lr_C = 1e-2
    opt.lr_M =1e-2
    opt.schedulerG_milestones = [20, 30, 40, 50]
    opt.schedulerC_milestones = [30, 45]
    opt.schedulerM_milestones = [10, 20]
    opt.schedulerG_lambda = 0.1
    opt.schedulerC_lambda = 0.1
    opt.schedulerM_lambda = 0.1
    opt.n_iters = 60
    opt.lambda_div = 1
    opt.lambda_norm = 100

    opt.target_label = 0
    opt.p_attack = 0.1
    opt.p_cross = 0.1
    opt.mask_density = 0.032
    opt.EPSILON = 1e-7

    opt.random_rotation = 10
    opt.random_crop = 5

    opt.input_height = args.img_size
    opt.input_width = args.img_size
    opt.input_channel = args.channel
    opt.batchsize = opt.batch_size

    path_model = os.path.join(
        opt.checkpoint, opt.dataset, opt.attack_mode, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset)
    )
    state_dict = torch.load(path_model)
    print("load C")
    # netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print("load G")
    netG = Generator(opt)
    netG.load_state_dict(state_dict["netG"])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)
    print("load M")
    netM = Generator(opt, out_channels=1)
    netM.load_state_dict(state_dict["netM"])
    netM.to(opt.device)
    netM.eval()
    netM.requires_grad_(False)

    # Prepare dataloader
    test_dl = iad_get_dataloader(opt, train=False)
    test_dl2 = iad_get_dataloader(opt, train=False)
    acc_clean, acc_bd = iad_eval(netC, netG, netM, test_dl, test_dl2, opt)
    return acc_clean, acc_bd


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


if __name__ == '__main__':
    main()
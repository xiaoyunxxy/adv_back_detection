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
parser.add_argument('--print-every', type=int, default=20, help='print results every few iterations')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output-dir', type=str, default='logs/models/')

parser.add_argument('--trigger-info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign'],
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--eps', type=float, default=0.4)
parser.add_argument('--steps', type=int, default=1)


args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
# os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    if args.trigger_info:
        trigger_info = torch.load(args.trigger_info, map_location=device)
    else:
        if args.poison_type == 'benign':
            trigger_info = None
        else:
            triggers = {'badnets': 'checkerboard_1corner',
                        'clean-label': 'checkerboard_4corner',
                        'blend': 'gaussian_noise'}
            trigger_type = triggers[args.poison_type]
            pattern, mask = poison_cifar.generate_trigger(trigger_type=trigger_type)
            trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}


    orig_train, clean_test = dataset_loader(args)

    sub_test, _ = random_split(dataset=clean_test, lengths=[args.val_frac, 1-args.val_frac], generator=torch.Generator().manual_seed(0))
    sub_train, _ = random_split(dataset=orig_train, lengths=[args.val_frac, 1-args.val_frac], generator=torch.Generator().manual_seed(0))

    print('number of samples in the sub_test: ', len(sub_test))

    # poisoned test set.
    if args.dataset == 'cifar10':
        poison_test = poison_cifar.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    elif args.dataset == 'gtsrb':
        poison_test = poison_gtsrb.add_predefined_trigger_gtsrb(clean_test, trigger_info)
    else:
        raise ValueError('Wrong dataset.')

    random_sampler = RandomSampler(data_source=sub_test, replacement=True,
                                   num_samples=args.print_every * args.batch_size)
    random_sampler_train = RandomSampler(data_source=sub_train, replacement=True,
                                   num_samples=args.print_every * args.batch_size)
    sub_train_loader = DataLoader(sub_train, batch_size=args.batch_size, shuffle=False, sampler=random_sampler_train, num_workers=8)
    sub_test_loader = DataLoader(sub_test, batch_size=args.batch_size, shuffle=False, sampler=random_sampler, num_workers=8)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=8)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=8)


    state_dict = torch.load(args.checkpoint, map_location=device)
    net = getattr(models, args.arch)(num_classes=args.num_classes, norm_layer=models.NoisyBatchNorm2d)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    poi_test_loss, poi_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print('Acc of the checkpoint, clean acc: {:.2f}, poison acc: {:.2f}'.format(cl_test_acc, poi_test_acc))

    non_perturb_acc = cl_test_acc

    parameters = list(net.named_parameters())
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.eps / args.steps)

    # Step 3: train backdoored models
    print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    for i in range(nb_repeat):
        start = time.time()
        lr = noise_optimizer.param_groups[0]['lr']
        perturbation_train(model=net, criterion=criterion, data_loader=sub_test_loader, noise_opt=noise_optimizer)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        end = time.time()
        train_loss = 0.0
        train_acc = 0.0
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))

    perturb_acc = cl_test_acc
    print('perturb_acc: ', perturb_acc, end='')
    print('    non_perturb_acc: ', non_perturb_acc)
  

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



def clip_noise(model, lower=-args.eps, upper=args.eps):
    params = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.reset(rand_init=rand_init, eps=args.eps)


def perturbation_train(model, criterion, noise_opt, data_loader):
    model.train()
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        # calculate the adversarial perturbation for neurons
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
                # clip_noise(model)



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



if __name__ == '__main__':
    main()

import argparse
import logging
import sys
import time
import math
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import shutil
from models import *
from utils import *
from tensorboardX import SummaryWriter
from utils_awp import AdvWeightPerturb

print("train at sp")
print("*"*100)


upper_limit, lower_limit = 1, 0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, BNeval=False):
    max_delta = torch.zeros_like(X).cuda()
    batch_size = len(X)

    if BNeval:
        model.eval()

    for _ in range(restarts):
        # initialize perturbation
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        else:
            raise ValueError

        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        iter_count = torch.zeros(y.shape[0])

        # craft adversarial examples
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()

            d = delta
            g = grad
            x = X
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g),
                                min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d
            delta.grad.zero_()

        max_delta = delta.detach()

    if BNeval:
        model.train()

    return max_delta, iter_count


def main():
    args = get_args()
    if args.fname == 'auto':
        names = get_auto_fname(args)
        args.fname = f'result_{args.prefix}/' + names
    else:
        args.fname = f'trained_{args.prefix}/' + args.fname

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    shutil.copy("./train_sp_awp.py", args.fname)
    shutil.copy("./utils_awp.py", args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    writer = SummaryWriter(os.path.join(args.fname, 'runs'))

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.data == "cifar10":
        print("use cifar10")
        trainset = torchvision.datasets.CIFAR10(
            root='~/datasets/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(
            root='~/datasets/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    elif args.data == "cifar100":
        print("use cifar100")
        trainset = torchvision.datasets.CIFAR100(
            root='~/datasets/cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR100(
            root='~/datasets/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Set perturbations
    epsilon = (args.epsilon / 255.)
    test_epsilon = (args.test_epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    test_pgd_alpha = (args.test_pgd_alpha / 255.)

    # Set models
    if args.model == 'VGG':
        model = VGG('VGG19')
    elif args.model == "ResNet34":
        model = ResNet34()
    elif args.model == 'ResNet18':
        num_class = 10 if args.data == "cifar10" else 100
        print(f"use resnet18  {num_class}")

        model = ResNet18(num_class=num_class)
        proxy = ResNet18(num_class=num_class)
    elif args.model == 'WideResNet':
        print("use wide resnet")
        model = WideResNet()
    else:
        raise ValueError("Unknown model")

    model = model.cuda()
    model.train()

    proxy = proxy.cuda()
    proxy.train()

    params = model.parameters()
    if args.optimizer == 'momentum':
        opt = torch.optim.SGD(params, lr=args.lr_max,
                              momentum=0.9, weight_decay=args.weight_decay)
        proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)

    awp_adversary = AdvWeightPerturb(
        model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=0.005)

    criterion = nn.CrossEntropyLoss()

    # Set lr schedulea
    if args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t < 75:
                return args.lr_max
            if args.lrdecay == 'base':
                if t < 90:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.

    best_test_robust_acc = 0
    if args.resume:
        print("-"*100)
        start_epoch = args.resume
        path = "~/model_last.pth"
        print(f"resuem from {path}")
        logger.info(f"resume form {path}")
        if ".pth" in path:
            model.load_state_dict(torch.load(path))
        else:
            model.load_state_dict(torch.load(
                os.path.join(path, f'model_last.pth')))
            best_test_robust_acc = torch.load(os.path.join(
                path, f'model_best.pth'))['test_robust_acc']
    else:
        start_epoch = 1

    logger.info(
        'Epoch \t lr \t Train Acc \t Train Robust Acc \t Test Acc \t Test Robust Acc')

    epochs = args.epochs

    for epoch in range(start_epoch, epochs+1):
        model.train()

        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0

        record_iter = torch.tensor([])

        for i, (X, y) in enumerate(trainloader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch)
            opt.param_groups[0].update(lr=lr)

            ##################### traing processure#####################################
            if args.attack == 'pgd':
                delta, iter_counts = attack_pgd(
                    model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, BNeval=args.BNeval)
                record_iter = torch.cat((record_iter, iter_counts))
                delta = delta.detach()

            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)

            ################### adversarial weigth perturbation############################

            adv_input = torch.clamp(
                X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
            adv_input.requires_grad = True

            clean_input = X
            clean_input.requires_grad = True

            if epoch > 10:
                awp = awp_adversary.calc_awp(
                    inputs_adv=adv_input, inputs_clean=clean_input, targets=y)
                awp_adversary.perturb(awp)

            adv_r_pred, adv_feature = model(adv_input)
            clean_logit, clean_feature = model(clean_input)

            robust_loss = criterion(adv_r_pred, y)

            normed_clean = F.normalize(clean_feature, dim=-1)
            matrix_clean = torch.mm(normed_clean, normed_clean.t())

            normed_feature = F.normalize(adv_feature, dim=-1)
            matrix_adv = torch.mm(normed_feature, normed_feature.t())

            diff = torch.exp(torch.abs(matrix_adv - matrix_clean))
            loss_sp = 10*torch.mean(diff)
            robust_loss += loss_sp

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            if epoch > 10:
                awp_adversary.restore(awp)

            #############################################################################################################

            clean_input = X
            clean_input.requires_grad = True
            output = model(clean_input)

            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            robust_output = adv_r_pred
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        print('Learning rate: ', lr)

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta, _ = attack_pgd(
                    model, X, y, test_epsilon, test_pgd_alpha, args.attack_iters, args.restarts, args.norm)
            delta = delta.detach()

            adv_input = torch.clamp(
                X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
            adv_input.requires_grad = True
            robust_output = model(adv_input)

            if isinstance(robust_output, tuple):
                robust_output = robust_output[0]
            robust_loss = criterion(robust_output, y)

            clean_input = X
            clean_input.requires_grad = True
            output = model(clean_input)

            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        if epoch >= 0:
            logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                        epoch, lr, train_acc/train_n, train_robust_acc/train_n, test_acc/test_n, test_robust_acc/test_n)

            # save tesnsorboard
            writer.add_scalar(f'train/nat_loss', train_loss /
                              train_n, global_step=epoch)
            writer.add_scalar(f'train/nat_acc', train_acc /
                              train_n*100, global_step=epoch)
            writer.add_scalar(f'train/robust_loss',
                              train_robust_loss/train_n, global_step=epoch)
            writer.add_scalar(f'train/robust_acc',
                              train_robust_acc/train_n*100, global_step=epoch)

            writer.add_scalar(f'test/nat_loss', test_loss /
                              test_n, global_step=epoch)
            writer.add_scalar(f'test/nat_acc', test_acc /
                              test_n*100, global_step=epoch)
            writer.add_scalar(f'test/robust_loss',
                              test_robust_loss/test_n, global_step=epoch)
            writer.add_scalar(f'test/robust_acc',
                              test_robust_acc/test_n*100, global_step=epoch)

            torch.save(model.state_dict(), os.path.join(
                args.fname, f'model_last.pth'))

            if epoch == 10 or epoch == 89 or epoch == 74 or epoch == 99 or epoch == 149:
                torch.save(model.state_dict(), os.path.join(
                    args.fname, f'model_{epoch}.pth'))
            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                    'state_dict': model.state_dict(),
                    'test_robust_acc': test_robust_acc/test_n,
                    'test_robust_loss': test_robust_loss/test_n,
                    'test_loss': test_loss/test_n,
                    'test_acc': test_acc/test_n,
                }, os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
    writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--affix', default=None, type=str)
    parser.add_argument('--prefix', default=None, type=str)
    parser.add_argument('--data', default='cifar10', type=str)
    parser.add_argument(
        '--data-dir', default='/home/lxb/datasets/cifar10', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=[
                        'superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='pgd', type=str,
                        choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--test_epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--test-pgd-alpha', default=2, type=float)
    parser.add_argument('--norm', default='l_inf',
                        type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)

    parser.add_argument('--weight_decay', default=2e-4,
                        type=float)  # weight decay

    parser.add_argument('--batch-size', default=128, type=int)  # batch size
    parser.add_argument('--lrdecay', default='base', type=str,
                        choices=['intenselr', 'base', 'looselr', 'lineardecay'])

    # whether use eval mode for BN when crafting adversarial examples
    parser.add_argument('--BNeval', action='store_true')
    parser.add_argument('--optimizer', default='momentum',
                        choices=['momentum', 'Nesterov', 'SGD_GC', 'SGD_GCC', 'Adam', 'AdamW'])

    return parser.parse_args()


def get_auto_fname(args):
    names = args.model + '_eps' + \
        str(args.epsilon) + '_bs' + str(args.batch_size) + \
        '_maxlr' + str(args.lr_max)
    if args.weight_decay != 5e-4:
        names = names + '_wd' + str(args.weight_decay)

    if args.lrdecay != 'base':
        names = names + '_' + args.lrdecay
    if args.BNeval:
        names = names + '_BNeval'
    if args.optimizer != 'momentum':
        names = names + '_' + args.optimizer
    if args.attack != 'pgd':
        names = names + '_' + args.attack
    names = names + f"_{args.epochs}" + f"_{args.data}"
    if args.affix:
        names = names + '_' + args.affix

    print('File name: ', names)
    return names


if __name__ == "__main__":
    main()

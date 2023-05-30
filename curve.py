import argparse
import copy
import logging
import os
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import *
from utils_plus import get_loaders


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon=8.0/255, alpha=2.0/255, attack_iters=20, restarts=1, random_init=True):
    model.eval()
    max_delta = torch.zeros_like(X).cuda()

    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()

        if random_init:
            delta.uniform_(-epsilon, epsilon)
            delta.data = clamp(delta, 0.0 - X, 1.0 - X)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output, _ = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta
            g = grad
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, 0.0 - X, 1.0 - X)
            delta.data = d
            delta.grad.zero_()
        max_delta = delta.detach()
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head', type=int, default=0)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument("--type", default="best", type=str)
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--data-dir', default='~/datasets/', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='result_cvpr_sp1/ResNet18_eps8_bs128_maxlr0.1_wd0.0002_BNeval_100_cifar10_base',
                        type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


def curve():
    args = get_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler()
        ])

    logger.info(args)

    _, test_loader = get_loaders(
        args.data_dir, args.batch_size, DATASET="cifar10")

    path = os.path.join(args.out_dir, f'model_{args.type}.pth')
    best_state_dict = torch.load(path)
    print(os.path.join(path))
    print(args.head)

    if args.model == "ResNet34":
        model = ResNet34(num_classes=10)
    elif args.model == 'ResNet18':
        model = ResNet18(num_classes=10)
    model = model.cuda()

    if 'state_dict' in best_state_dict.keys():
        model.load_state_dict(best_state_dict['state_dict'])
    else:
        model.load_state_dict(best_state_dict)
    model.float()
    model.eval()

    len = 0
    clean_acc_list = []
    adv_acc_list = []
    matrix_diff_list = []
    for indexx, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()

        x_adv = x + attack_pgd(model, x, y)
        len += y.size(0)

        clean_out, clean_feature = model(x)
        adv_out, adv_feature = model(x_adv)
        clean_feature = F.normalize(clean_feature, dim=-1)
        clean_matrix = torch.mm(
            clean_feature, clean_feature.t()).detach().cpu().numpy()
        adv_feature = F.normalize(adv_feature, dim=-1)
        adv_matrix = torch.mm(adv_feature, adv_feature.t()
                              ).detach().cpu().numpy()

        matrix_diff = np.mean(np.exp(np.abs(clean_matrix - adv_matrix)))

        adv_acc = (adv_out.max(1)[1] == y)
        clean_acc = (clean_out.max(1)[1] == y)

        clean_acc_list.append(clean_acc.float().mean().item())
        adv_acc_list.append(adv_acc.float().mean().item())
        matrix_diff_list.append(matrix_diff)

    plt.scatter(matrix_diff_list, adv_acc_list)
    plt.savefig(f"{args.head}.png")

    print(clean_acc_list)
    print(adv_acc_list)
    print("test over")


if __name__ == "__main__":
    curve()

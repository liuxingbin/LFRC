#import apex.amp as amp
import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pdb

upper_limit, lower_limit = 1, 0

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


def normalize(X):
    return X


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size, DATASET='cifar10'):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if DATASET == 'cifar10':
        test_dataset = datasets.CIFAR10(
            dir_+'/cifar10', train=False, transform=test_transform, download=True)
    elif DATASET == 'cifar100':
        test_dataset = datasets.CIFAR100(
            dir_+'/cifar100', train=False, transform=test_transform, download=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    return None, test_loader


def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2]
                   * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=False, random_init=True):
    max_delta = torch.zeros_like(X).cuda()

    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()

        if random_init:
            delta.uniform_(-epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta
            g = grad
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X, upper_limit - X)
            delta.data = d
            delta.grad.zero_()
        max_delta = delta.detach()
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, eps=8, step=2, use_CWloss=False, random_init=True, black_model=None):
    epsilon = eps / 255.
    alpha = step / 255.
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    attack_list = []
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()

        if black_model:
            pgd_delta = attack_pgd(black_model, X, y, epsilon, alpha, attack_iters,
                                   restarts, use_CWloss=use_CWloss, random_init=random_init)
        else:
            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters,
                                   restarts, use_CWloss=use_CWloss, random_init=random_init)

        noise_x = normalize(X + pgd_delta)
        for index in range(20):
            torchvision.utils.save_image(
                noise_x[index], '../{}.png'.format(index))

        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            attack_list.extend((output.max(1)[1]).cpu().numpy())

    return pgd_loss/n, pgd_acc/n, attack_list


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    clean_list = []
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(normalize(X))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            clean_list.extend(y.cpu().numpy())
    return test_loss/n, test_acc/n, clean_list

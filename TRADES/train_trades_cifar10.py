#!/usr/bin/python
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import sys
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter, writer

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss
import shutil
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=8.0/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2.0/255,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./results/test_time',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model_dir
sys.stdout = Logger(model_dir + "/train.log")

writer = SummaryWriter(os.path.join(model_dir, "runs"))
shutil.copy("./train_trades_cifar10.py", model_dir)
shutil.copy("./trades.py", model_dir)

print(str(args))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='~/datasets/cifar10', train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(
    root='~/datasets/cifar10', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# trainset = torchvision.datasets.CIFAR100(root='~/datasets/cifar100', train=True, download=True, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
# testset = torchvision.datasets.CIFAR100(root='~/datasets/cifar100', train=False, download=True, transform=transform_test)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           epoch=epoch,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output,
                                          target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output,
                                         target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 15200:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


upper_limit = torch.tensor(1.0).cuda()
lower_limit = torch.tensor(0.0).cuda()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters):

    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon.item(), epsilon.item())
    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()

        d = delta
        g = grad
        d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
        d = clamp(d, lower_limit - X, upper_limit - X)
        delta.data = d
        delta.grad.zero_()
    return delta


def evaluate_pgd(test_loader, model, attack_iters=10, step=2):
    epsilon = torch.tensor(8 / 255.).cuda()
    alpha = torch.tensor(step / 255.).cuda()
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def main():
    model = ResNet18(num_class=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    best_acc = 0.00

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        start_time = time.time()
        lr = adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        print("time\t", time.time() - start_time)

        # evaluation on natural examples
        print('================================================================')
        test_loss, test_acc = evaluate_pgd(test_loader, model)
        print(
            f"epoch:\t{epoch}\tlr:\t{lr}\ttest loss:\t{test_loss}\ttest_acc:\t{test_acc}")
        print('================================================================')

        # save checkpoint
        if epoch == 74 or epoch == 99:
            torch.save(model.state_dict(), os.path.join(
                model_dir, f"{epoch}.pth"))
        torch.save(model.state_dict(), os.path.join(
            model_dir, "model_last.pth"))
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(
                model_dir, "model_best.pth"))
        writer.add_scalar(f"test_acc", test_acc, global_step=epoch)

    writer.close()


if __name__ == '__main__':
    main()

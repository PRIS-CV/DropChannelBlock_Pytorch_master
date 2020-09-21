'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
import shutil

import time
import torch
import logging
import argparse
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from util.cutout import Cutout
from util.utils import AverageMeter

import cv2

#from utils import progress_bar
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='PyTorch ResNet Baseline Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--exp_name', default='baseline_birds', type=str, help='store name')
parser.add_argument('--gpu', default='3', type=str, help='gpu')
parser.add_argument('--seed', default=2020, type=int, help='seed')

args = parser.parse_args()
logging.info(args)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

exp_dir = os.path.join("results", args.exp_name) 

nb_epoch = 200
init_lr = 0.002
PRINT_FREQ = 50

try:
    os.stat('visual')
except:
    os.makedirs('visual')
save_dir = os.path.join('visual', args.exp_name)
try:
    os.stat(save_dir)
except:
    os.makedirs(save_dir)

try:
    os.stat(exp_dir)
except:
    os.makedirs(exp_dir)
logging.info("OPENING " + exp_dir + '/results_train.csv')
logging.info("OPENING " + exp_dir + '/results_test.csv')

results_train_file = open(exp_dir + '/results_train.csv', 'w')
results_train_file.write('epoch, train_acc, train_loss\n')
results_train_file.flush()

results_test_file = open(exp_dir + '/results_test.csv', 'w')
results_test_file.write('epoch, test_acc\n')
results_test_file.flush()

use_cuda = torch.cuda.is_available()

#Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale((550, 550)),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # Cutout(n_holes=1, length=112),
])

transform_test = transforms.Compose([
    transforms.Scale((550, 550)),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

trainset    = torchvision.datasets.ImageFolder(root='/mnt/2/donggua/Birds2/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(root='/mnt/2/donggua/Birds2/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)


# # Model
print('==> Building model..')

from model.ResNetBase import ResNetBase
shutil.copy2(os.path.join('model', 'ResNetBase.py'), exp_dir)
net = ResNetBase(model_name="resnet50", num_classes=200, pretrained=True)

if use_cuda:
    net.cuda()
    cudnn.benchmark = True
    # device = torch.device("cuda:0,1")
    # net.to(device)
    # netp = torch.nn.DataParallel(net, device_ids=[0,1])

criterion = nn.CrossEntropyLoss()

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        out, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel = model(inputs)
        
        # compute gradient and do update step
        loss = criterion(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        _, predicted = torch.max(out.data, 1)
        correct = predicted.eq(targets.data).cpu().sum().item()
        acc.update(100.*float(correct)/inputs.size(0), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inputs.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logging.info(msg)

            vis_input = torchvision.utils.make_grid(inputs, nrow=8, padding=2,normalize=True)
            cv2.imwrite(os.path.join(save_dir, 'train_inputs_{}.jpg'.format(i)), (vis_input*255).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8))
            vis_heatmap_all = torchvision.utils.make_grid(heatmap_all, nrow=8, padding=2,normalize=True)
            cv2.imwrite(os.path.join(save_dir, 'train_heatmap_all_{}.jpg'.format(i)), (vis_heatmap_all*255).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8))
            vis_heatmap_remain = torchvision.utils.make_grid(heatmap_remain, nrow=8, padding=2,normalize=True)
            cv2.imwrite(os.path.join(save_dir, 'train_heatmap_remain_{}.jpg'.format(i)), (vis_heatmap_remain*255).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8))
            vis_heatmap_drop = torchvision.utils.make_grid(heatmap_drop, nrow=8, padding=2,normalize=True)
            cv2.imwrite(os.path.join(save_dir, 'train_heatmap_drop_{}.jpg'.format(i)), (vis_heatmap_drop*255).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8))
            vis_select_channel = torchvision.utils.make_grid(select_channel, nrow=8, padding=2,normalize=True)
            cv2.imwrite(os.path.join(save_dir, 'train_select_channel_{}.jpg'.format(i)), (vis_select_channel*255).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8))

            vis_all_channel = torchvision.utils.make_grid(all_channel, nrow=8, padding=2,normalize=True)
            cv2.imwrite(os.path.join(save_dir, 'train_all_channel_{}.jpg'.format(i)), (vis_all_channel*255).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8))

    results_train_file.write('%d, %.4f, %.4f\n' % (epoch, acc.avg, losses.avg))
    results_train_file.flush()
 

def validate(val_loader, model, epoch):
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            # compute output
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            out, heatmap, _, _, _, _ = model(inputs)

            # measure accuracy
            _, predicted = torch.max(out.data, 1)
            correct = predicted.eq(targets.data).cpu().sum().item()
            acc.update(100.*float(correct)/inputs.size(0), inputs.size(0))

            if i % PRINT_FREQ == 0:
                vis_input = torchvision.utils.make_grid(inputs, nrow=8, padding=2,normalize=True)
                cv2.imwrite(os.path.join(save_dir, 'test_inputs_{}.jpg'.format(i)), (vis_input*255).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8))
                vis_heatmap = torchvision.utils.make_grid(heatmap, nrow=8, padding=2,normalize=True)
                cv2.imwrite(os.path.join(save_dir, 'test_heatmap_{}.jpg'.format(i)), (vis_heatmap*255).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8))


    msg = 'Epoch: [{0}][{1}/{2}]\t' \
        'Accuracy3 {acc.val:.3f} ({acc.avg:.3f})'.format(
            epoch, i, len(val_loader), acc=acc)
    logging.info(msg)
    results_test_file.write('%d, %.4f\n' % (epoch, acc.avg))
    results_test_file.flush()

    return acc.avg

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch)) 
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( lr / 2 * cos_out)

base_params = list(map(id, net.resnet.parameters()))
logits_params = filter(lambda p: id(p) not in base_params, net.parameters())
optimizer = optim.SGD([
                        {'params': logits_params, 'lr': init_lr},
                        {'params': net.resnet.parameters(), 'lr': init_lr / 10}
                     ], 
                      momentum=0.9, weight_decay=5e-4)

max_val_acc = 0
for epoch in range(0, nb_epoch):
    optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, init_lr)
    optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, init_lr) / 10
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    train(trainloader, net, criterion, optimizer, epoch)
    if epoch < 5 or epoch >= 80:
        val_acc = validate(testloader, net, epoch)
    if val_acc > max_val_acc:
        max_val_acc = val_acc
    print('max_val_acc=', max_val_acc)

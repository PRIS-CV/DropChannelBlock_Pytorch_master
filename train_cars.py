'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import torch
import logging
import argparse
import torchvision
#from models import *
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
# from my_pooling import my_MaxPool2d
import torchvision.transforms as transforms

from util.visualize import * 
import cv2

#from utils import progress_bar
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--exp_name', default='dc_tmp', type=str, help='store name')
parser.add_argument('--gpu', default='3', type=str, help='gpu')
parser.add_argument('--seed', default=2020, type=int, help='seed')

args = parser.parse_args()
logging.info(args)
#from rate import CyclicScheduler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# vis = Visualizer(env="cars_new_adj_matrix_0406_2cls", port=8097, server="http://localhost")

store_name = os.path.join("results", args.exp_name) 
# setup output
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir = store_name 


nb_epoch = 100


try:
    os.stat(exp_dir)
except:
    os.makedirs(exp_dir)
logging.info("OPENING " + exp_dir + '/results_train.csv')
logging.info("OPENING " + exp_dir + '/results_test.csv')


results_train_file = open(exp_dir + '/results_train.csv', 'w')
results_train_file.write('epoch, train_acc,train_loss\n')
results_train_file.flush()

results_test_file = open(exp_dir + '/results_test.csv', 'w')
results_test_file.write('epoch, test_acc,test_loss\n')
results_test_file.flush()



use_cuda = torch.cuda.is_available()

# def plot_img(adjs, mode='train'):
#     with torch.no_grad():
#         for i in range(1):
#             mask = adjs[i].data.cpu().numpy()
#             # mask = np.exp(10 * mask)
#             img_mask = (255.0 * (mask - np.min(mask)) / (np.max(mask) - np.min(mask))+1e-8).astype(np.uint8)
#             # img_mask = (255.0 * mask).astype(np.uint8)
#             # img_mask = cv2.resize(img_mask, dsize=(512, 512))
#             vis.img('%s_img_%d' % (mode, i), img_mask)

#Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale((448,448)),
    transforms.RandomCrop(448, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Scale((448,448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



#trainset    = torchvision.datasets.ImageFolder(root='./train', transform=transform_train)
trainset    = torchvision.datasets.ImageFolder(root='/data/dingyifeng/StandCars/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)


#testset = torchvision.datasets.ImageFolder(root='./test', transform=transform_test)
testset = torchvision.datasets.ImageFolder(root='/data/dingyifeng/StandCars/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)


# # Model

print('==> Building model..')


from model.resnet_dc import resnet18, resnet50 
from model.vgg_dc import vgg16, vgg19 

net = resnet50(num_classes=196)
pretrained_path = "/home/dingyifeng/.torch/models/resnet50-19c8e357.pth"

# net = vgg19(num_classes=196)
# pretrained_path = "/home/dingyifeng/.torch/models/vgg19_bn-c79401a0.pth"

if pretrained_path:
    logging.info('load pretrained backbone')
    net_dict = net.state_dict()
    pretrained_dict = torch.load(pretrained_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

if use_cuda:
    net.cuda()
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
# scheduler = CyclicScheduler(base_lr=0.00001, max_lr=0.01, step=2050., mode='triangular2', gamma=1., scale_fn=None, scale_mode='cycle') ##exp_range ##triangular2

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    flag = 1

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #optimizer.param_groups[0]['lr'] = scheduler.get_rate()
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)


        loss = criterion(outputs, targets)


        loss.backward()
        optimizer.step()

        train_loss += loss.data

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    train_acc = 100.*float(correct)/total
    train_loss = train_loss/(idx+1)
    logging.info('Iteration %d, train_acc = %.5f,train_loss = %.6f' % (epoch, train_acc,train_loss))
    results_train_file.write('%d, %.4f,%.4f\n' % (epoch, train_acc,train_loss))
    results_train_file.flush()
    return train_acc, train_loss

def test(epoch):
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        test_acc = 100.*float(correct)/total
        test_loss = test_loss/(idx+1)
        logging.info('Iteration %d, test_acc = %.4f,test_loss = %.4f' % (epoch, test_acc,test_loss))
        results_test_file.write('%d, %.4f,%.4f\n' % (epoch, test_acc,test_loss))
        results_test_file.flush()

    return test_acc
 



def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  )) 
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.001 / 2 * cos_out)



optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epoch)

# optimizer = optim.SGD([
#                         {'params': nn.Sequential(*list(net.children())[7:]).parameters(),   'lr': 0.001},
#                         {'params': nn.Sequential(*list(net.children())[:7]).parameters(),   'lr': 0.0001}
                        
#                      ], 
#                       momentum=0.9, weight_decay=5e-4)


max_val_acc = 0
for epoch in range(0, nb_epoch):
    scheduler.step(epoch)
    # optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch) / 10
    # optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    train(epoch)
    val_acc = test(epoch)
    if val_acc >max_val_acc:
        max_val_acc = val_acc
        # torch.save(net.state_dict(), store_name+'.pth')

# torch.cuda.empty_cache()
# os.system('python /home/dingyifeng/utils/train.py --gpu {}'.format(args.gpu)) 
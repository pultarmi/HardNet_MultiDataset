#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this code, please cite
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin
"""

from __future__ import division, print_function
from copy import deepcopy
import random, cv2, copy, os, sys, torch, argparse, PIL
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from EvalMetrics import ErrorRateAt95Recall#, ErrorRateFDRAt95Recall, convertFDR2FPR, convertFPR2FDR
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization
from W1BS import w1bs_extract_descs_and_save
# from Utils import adjust_learning_rate, HardNet, TripletPhotoTour, create_optimizer, L2Norm, cv2_scale, np_reshape32, np_reshape64, str2bool, test, HardNetPS
from Utils import *
from HandCraftedModules import get_WF_from_string
import torch.nn as nn
import torch.utils.data as data
from os import path
from utils import send_email, get_last_checkpoint


class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum()) / input.size(0)

parser = argparse.ArgumentParser(description='PyTorch HardNet')
parser.add_argument('--dataroot', type=str, default='../Process_DS/Datasets/', help='path to dataset')
parser.add_argument('--model-dir', default='models/', help='folder to output model checkpoints')
parser.add_argument('--training-set', default='FullBrown6', help='Other options: notredame, yosemite')
parser.add_argument('--loss', default='triplet_margin', help='Other options: softmax, contrastive')
parser.add_argument('--batch-reduce', default='min', help='Other options: average, random, random_global, L2Net')
# parser.add_argument('--num-workers', default=1, help='Number of workers to be created') # but is separate to amos/6Brown
# parser.add_argument('--pin-memory', type=bool, default=True, help='')
parser.add_argument('--decor', type=str2bool, default=False, help='L2Net decorrelation penalty')
parser.add_argument('--anchorave', type=str2bool, default=False, help='anchorave')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019, help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209, help='std of train dataset for normalization')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=bool, default=True, help='turns on anchor swap')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS', help='input batch size for training (default: 1024)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='BST', help='input batch size for testing (default: 1024)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N', help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN', help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--gor', type=str2bool, default=False, help='use gor')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA', help='gor parameter')
parser.add_argument('--act-decay', type=float, default=0, help='activity L2 decay, default 0')
parser.add_argument('--lr', type=float, default=20.0, metavar='LR', help='learning rate (default: 10.0)')
parser.add_argument('--fliprot', type=str2bool, default=True, help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD', help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str, metavar='OPT', help='The optimizer to use (default: SGD)')
parser.add_argument('--n-patch-sets', type=int, default=30000, help='How many patch sets to generate. 300k is ~ 6000 per image seq for HPatches')
parser.add_argument('--id', type=int, default=0, help='id')
parser.add_argument('--tower-dataset', type=str, default='', help='path to HSequences-like dataset (one dir per seq with homographies)')

# parser.add_argument('--n-positives', type=int, default=2, help='How many positive patches to generate per point. Max number == # images in seq.')

parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI', help='how many batches to wait before logging training status')
parser.add_argument('--regen-each-iter', type=str2bool, default=False, help='Regenerate keypoints each iteration, default True')

# parser.add_argument('--new-patches', type=int, default=0, help='Use new method to generate patches')
parser.add_argument('--masks-dir', type=str, default=None, help='you can specify where masks are saved')
parser.add_argument('--mark-patches-dir', type=str, default=None, help='you can specify where masks are saved')
parser.add_argument('--cams-in-batch', type=int, default=0, help='you can specify where masks are saved')

parser.add_argument('--patch-gen', type=str, default='oneRes', help='options: oneImg, sumImg, watchGood')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--PS', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true', help='verbal')

parser.add_argument('--spx', type=float, default=0, help='sx')
parser.add_argument('--spy', type=float, default=0, help='sy')

parser.add_argument('--weight-function', type=str, default='Hessian',
                    help='Keypoints are generated with probability ~ weight function. If None (default), then uniform sampling. Variants: Hessian, HessianSqrt, HessianSqrt4, None')
args = parser.parse_args()

txt = []
txt += ['PS:'+str(args.n_patch_sets)+'PP']
txt += ['WF:'+args.weight_function]
txt += ['PG:'+args.patch_gen]
txt += ['masks:'+str(int(args.masks_dir is not None))]
txt += ['spx:'+str(args.spx)]
txt += ['spy:'+str(args.spy)]
split_name = '_'.join(txt)

txt = []
txt += [path.basename(args.tower_dataset).lower()]
txt += ['id:'+str(args.id)]
txt += ['TrS:'+args.training_set]
txt += [split_name]
txt += [args.batch_reduce]
txt += ['tps:'+str(args.n_triplets)]
txt += ['camsB:'+str(args.cams_in_batch)]
txt += ['ep:'+str(args.epochs)]
if args.gor: txt += ['gor_alpha{:1.1f}'.format(args.alpha)]
if args.anchorswap: txt += ['as']
save_name = '_'.join([str(c) for c in txt])

# triplet_flag = (args.batch_reduce == 'random_global') or args.gor

test_dataset_names = []
# test_dataset_names += ['liberty']
# test_dataset_names += ['notredame']
# test_dataset_names += ['yosemite']
# test_dataset_names += ['amos_10K']

cudnn.benchmark = True
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def transform(img):
    img = (img.numpy()) / 255.0
    return transforms.Compose([
        transforms.Lambda(cv2_scale),
        transforms.Lambda( lambda x: np.reshape(x, (32, 32, 1)) ),
        transforms.ToTensor(),
    ])(img)
def t(img):
    img = transforms.Compose([
        transforms.Lambda( lambda x: np.reshape(x, (1, 64, 64)) ),
        transforms.ToPILImage(),
        transforms.RandomAffine(25, scale=(0.8, 1.4), shear=25, resample=PIL.Image.BICUBIC),
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.9, 1.10)),
        transforms.ToTensor(),
    ])(img)
    return img.type(torch.float64)
normal_transform={'default':transform}
# easy_transform={'e1':t, 'e2':t, 'e3':t, 'default':transform}

transform_AMOS = transforms.Compose([transforms.ToPILImage(),
                                     transforms.RandomAffine(25, scale=(0.8, 1.4), shear=25, resample=PIL.Image.BICUBIC),
                                     transforms.CenterCrop(64),
                                     transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.9, 1.10)),
                                     transforms.ToTensor()])

def create_loaders():
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loaders = [
        {'name': name,
         'dataloader': torch.utils.data.DataLoader(
            TripletPhotoTour(train=False, batch_size=args.test_batch_size, n_triplets = 1000, root=path.join(args.dataroot, 'Test'), name=name, download=False, transform=transform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)} for name in test_dataset_names]

    return test_loaders

def train(train_loader, model, optimizer, epoch, load_triplets=False, WBSLoader=None):
    model.train()

    # train_loader.prepare_epoch()
    pbar = tqdm(enumerate(train_loader))
    # if WBSLoader is not None:
    #     WBSiter = iter(WBSLoader)
    for batch_idx, data in pbar:
        if load_triplets:
            data_a, data_p, data_n = data
        else:
            data_a, data_p = data

            # if WBSLoader is not None and batch_idx < len(WBSLoader):
            #     pom_a, pom_p = next(WBSiter)
            #     data_a = torch.cat((data_a.float(), pom_a.float()))
            #     data_p = torch.cat((data_p.float(), pom_p.float()))

        def fce(data_a, data_p, change_lr=True):
            data_a, data_p = Variable(data_a.cuda()), Variable(data_p.cuda())
            out_a, out_p = model(data_a), model(data_p)

            if load_triplets:
                data_n = data_n.cuda()
                data_n = Variable(data_n)
                out_n = model(data_n)

            if args.batch_reduce == 'L2Net':
                loss = loss_L2Net( out_a, out_p, anchor_swap=args.anchorswap, margin=args.margin, loss_type=args.loss )
            elif args.batch_reduce == 'random_global':
                loss = loss_random_sampling( out_a, out_p, out_n, margin=args.margin, anchor_swap=args.anchorswap, loss_type=args.loss )
            else:
                loss = loss_HardNet( out_a, out_p, margin=args.margin, anchor_swap=args.anchorswap, anchor_ave=args.anchorave, batch_reduce=args.batch_reduce, loss_type=args.loss )
                loss = loss.mean()

            if args.decor:
                loss += CorrelationPenaltyLoss()(out_a)

            if args.gor:
                loss += args.alpha * global_orthogonal_regularization(out_a, out_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if change_lr:
                adjust_learning_rate( optimizer, args.lr, args.batch_size, args.n_triplets, args.epochs )
            if batch_idx % args.log_interval == 0:
                pbar.set_description( 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_a), len(train_loader)*len(data_a), 100. * batch_idx / len(train_loader), loss.item()) )

        fce(data_a, data_p)

    os.makedirs(os.path.join(args.model_dir, save_name), exist_ok=True)
    save_path = '{}{}/checkpoint_{}.pth'.format(args.model_dir, save_name, epoch)
    torch.save( {'epoch': epoch + 1, 'state_dict': model.state_dict()}, save_path)
    print('saving to: {}'.format(save_path))
    print_lr(optimizer)

def print_lr(optimizer):
    for group in optimizer.param_groups:
        print( "Learning rate: " +str(group['lr']) )
        return

def main(train_loader, test_loaders, model):
    print('\nparsed options:\n{}\n'.format(vars(args)))
    model.cuda()

    optimizer1 = create_optimizer(model.features, args.lr, args.optimizer, args.wd)

    if args.resume: # optionally resume from a checkpoint
        p1 = args.resume
        p2 = os.path.join(args.model_dir, args.resume)
        path_resume = None

        if os.path.isfile(p1):
            # print('=> no checkpoint found at {}'.format(os.path.join(os.getcwd(), args.resume)))
            # path_resume = os.path.join(args.model_dir, args.resume)
            path_resume = p1
        elif os.path.isfile(p2):
            # print('=> no checkpoint found at {}'.format(os.path.join(os.getcwd(), args.resume)))
            # path_resume = os.path.join(args.model_dir, args.resume)
            path_resume = p2
        elif os.path.exists(p2):
            print('searching dir')
            path_resume = os.path.join(p2, get_last_checkpoint(p2))
            # print('path exists, last checkpoint found: {}'.format(path_resume))

        if path_resume is not None:
            print('=> loading checkpoint {}'.format(path_resume))
            if args.PS:
                model = HardNetPS()
                checkpoint = torch.load(path_resume)
                model.load_state_dict(checkpoint)
                model.cuda()
            else:
                checkpoint = torch.load(path_resume)
                try:
                    args.start_epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['state_dict'])
                except:
                    print('loading subset of weights')
                    pom = model.state_dict()
                    pom.update(checkpoint)
                    model.load_state_dict(pom)
                try:
                    optimizer1.load_state_dict(checkpoint['optimizer'])
                except:
                    print('optimizer not loaded')
        else:
            print('=> no checkpoint found')
            # print('=> no checkpoint found at {}'.format(path_resume))

    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        train(train_loader, model, optimizer1, epoch, False, WBSLoader=None)

        for test_loader in test_loaders:
            test( test_loader['dataloader'], model, epoch, test_loader['name'], args )

if __name__ == '__main__':
    model = HardNet()
    test_loaders = create_loaders()

    datasets_path = path.join(args.dataroot, 'Train')
    datasets_path = sorted([os.path.join(datasets_path, dataset) for dataset in os.listdir(datasets_path) if '.pt' in dataset])
    DSs = []
    for i in range(len(datasets_path)):
        DSs += [One_DS(Args_Brown(datasets_path[i], 5, True, normal_transform), group_id=[i])]
    DSs += [One_DS(Args_AMOS(args.tower_dataset, split_name, args.n_patch_sets, get_WF_from_string(args.weight_function), args.batch_size, True, transform_AMOS,
                             args.patch_gen, args.cams_in_batch), group_id=list(range(len(datasets_path))))]
        # DSs += [One_DS(datasets_path[0], 5, True, easy_transform, FORMAT.Brown, group_id=[0,1])]
        # DSs += [One_DS(datasets_path[0], 5, True, easy_transform, FORMAT.Brown, group_id=[0])]
        # DSs += [One_DS(datasets_path[0], 5, True, easy_transform, FORMAT.AMOS, group_id=[1])]

    wrapper = DS_wrapper(DSs, args.n_triplets, args.batch_size)
    print('----------------\nsplit_name: {}'.format(split_name))
    print('save_name: {}'.format(save_name))
    main(wrapper, test_loaders, model)
    print('Train end, saved: {}'.format(save_name))
    send_email(recipient='milan.pultar@gmail.com', ignore_host='milan-XPS-15-9560') # useful fo training, change the recipient address for yours or comment this out
#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this code, please cite:

@article{HardNet2017,
    author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
    year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin

@article{HardNetAMOS2019,
    author = {Milan Pultar, Dmytro Mishkin, Jiri Matas},
    title = "{Leveraging Outdoor Webcams for Local Descriptor Learning}",
    year = 2019,
    month = feb,
    booktitle = {Proceedings of CVWW 2019}
}
"""
from __future__ import division, print_function
import torch.nn.init
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from Utils import *
from HandCraftedModules import get_WF_from_string
import torch.nn as nn
import torch.utils.data as data
from os import path
from utils import send_email, get_last_checkpoint


parser = argparse.ArgumentParser(description="PyTorch HardNet")
parser.add_argument("--model-dir", default="models/", help="folder to output model checkpoints")
parser.add_argument("--name", default="", help="Other options: notredame, yosemite")
parser.add_argument("--loss", default="triplet_margin", help="Other options: softmax, contrastive")
parser.add_argument("--batch-reduce", default="min", help="Other options: average, random, random_global, L2Net")
parser.add_argument("--anchorave", type=str2bool, default=False, help="anchorave")
parser.add_argument("--imageSize", type=int, default=32, help="the height / width of the input image to network")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("--epochs", type=int, default=10, metavar="E", help="number of epochs to train (default: 10)")
parser.add_argument("--anchorswap", type=bool, default=True, help="turns on anchor swap")
parser.add_argument("--batch-size", type=int, default=1024, metavar="BS", help="input batch size for training (default: 1024)")
parser.add_argument("--test-batch-size", type=int, default=128, metavar="BST", help="input batch size for testing (default: 1024)")
parser.add_argument("--n-triplets", type=int, default=5000000, metavar="N", help="how many triplets will generate from the dataset")
parser.add_argument("--margin", type=float, default=1.0, metavar="MARGIN", help="the margin value for the triplet loss function (default: 1.0")
parser.add_argument("--lr", type=float, default=20.0, metavar="LR", help="learning rate (default: 10.0)")
parser.add_argument("--fliprot", type=str2bool, default=True, help="turns on flip and 90deg rotation augmentation")
parser.add_argument("--wd", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)")
parser.add_argument("--optimizer", default="sgd", type=str, metavar="OPT", help="The optimizer to use (default: SGD)")
parser.add_argument("--n-patch-sets", type=int, default=30000, help="How many patch sets to generate. 300k is ~ 6000 per image seq for HPatches")
parser.add_argument("--id", type=int, default=0, help="id")

parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
parser.add_argument("--log-interval", type=int, default=1, metavar="LI", help="how many batches to wait before logging training status")
parser.add_argument("--regen-each-iter", type=str2bool, default=False, help="Regenerate keypoints each iteration, default True")

parser.add_argument("--mark-patches-dir", type=str, default=None, help="you can specify where masks are saved")
parser.add_argument("--cams-in-batch", type=int, default=0, help="you can specify where masks are saved")

parser.add_argument("--patch-gen", type=str, default="oneRes", help="options: oneImg, sumImg, watchGood")
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--PS", default=False, action="store_true")
parser.add_argument("--debug", default=False, action="store_true", help="verbal")

parser.add_argument("--spx", type=float, default=0, help="sx")
parser.add_argument("--spy", type=float, default=0, help="sy")

parser.add_argument(
    "--weight-function",
    type=str,
    default="Hessian",
    help="Keypoints are generated with probability ~ weight function. If None (default), then uniform sampling. Variants: Hessian, HessianSqrt, HessianSqrt4, None",
)
args = parser.parse_args()

txt = []
txt += ["PS:" + str(args.n_patch_sets) + "PP"]
txt += ["WF:" + args.weight_function]
txt += ["PG:" + args.patch_gen]
# txt += ['masks:'+str(int(args.masks_dir is not None))]
txt += ["spx:" + str(args.spx)]
txt += ["spy:" + str(args.spy)]
split_name = "_".join(txt)

txt = []
# txt += [path.basename(args.tower_dataset).lower()]
txt += ["id:" + str(args.id)]
txt += ["TrS:" + args.name]
# txt += ['TrS:'+args.training_set]
txt += [split_name]
txt += [args.batch_reduce]
txt += ["tps:" + str(args.n_triplets)]
txt += ["camsB:" + str(args.cams_in_batch)]
txt += ["ep:" + str(args.epochs)]
if args.anchorswap:
    txt += ["as"]
save_name = "_".join([str(c) for c in txt])


cudnn.benchmark = True
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
transform_test = transforms.Compose([
            #transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()])

transform_train_1 = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(25, scale=(0.8, 1.4), shear=25, resample=PIL.Image.BILINEAR),
            transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.9, 1.10)),
            transforms.ToTensor(),
        ]
    )


default_transform = {"default": transform_train_1}
# easy_transform={'e1':t, 'e2':t, 'e3':t, 'default':transform}

transform_AMOS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomAffine(25, scale=(0.8, 1.4), shear=25, resample=PIL.Image.BILINEAR),
        transforms.CenterCrop(64),
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.9, 1.10)),
        transforms.ToTensor(),
    ]
)


def get_test_loaders():
    kwargs = {"num_workers": 4, "pin_memory": True}
    test_loaders = [
        {
            "name": name,
            "dataloader": torch.utils.data.DataLoader(
                TripletPhotoTour(train=False, batch_size=args.test_batch_size, n_triplets=1000, root=path.join("Datasets"), name=name, download=True, transform=transform_test),
                batch_size=args.test_batch_size,
                shuffle=False,
                **kwargs
            ),
        }
        for name in ["liberty", "notredame", "yosemite", "liberty_harris", "notredame_harris", "yosemite_harris"]
    ]

    return test_loaders


def train(train_loader, model, optimizer, epoch, load_triplets=False, WBSLoader=None):
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        data_a, data_p = data
        data_a = data_a.cuda()
        data_p = data_p.cuda()
        out_a = model(data_a)
        out_p = model(data_p)
        loss = loss_HardNet(out_a, out_p, margin=args.margin, anchor_swap=args.anchorswap, anchor_ave=False, batch_reduce=args.batch_reduce, loss_type=args.loss)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pom = adjust_learning_rate(optimizer, args.lr, args.batch_size, args.n_triplets, args.epochs)
        if pom < 0:  # just to be sure - never ascend
            break
        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data_a), len(train_loader) * len(data_a), 100.0 * batch_idx / len(train_loader), loss.item())
            )

    os.makedirs(os.path.join(args.model_dir, save_name), exist_ok=True)
    # save_path = '{}{}/checkpoint_{}.pth'.format(args.model_dir, save_name, epoch)
    save_path = os.path.join(args.model_dir, save_name, "checkpoint_{}.pth".format(epoch))
    torch.save({"epoch": epoch + 1, "state_dict": model.state_dict()}, save_path)
    print("saving to: {}".format(save_path))
    print_lr(optimizer)


def print_lr(optimizer):
    for group in optimizer.param_groups:
        print("Learning rate: " + str(group["lr"]))
        return


def main(train_loader, test_loaders, model):
    print("\nparsed options:\n{}\n".format(vars(args)))
    model.cuda()
    optimizer1 = create_optimizer(model, args.lr, args.optimizer, args.wd)
    if args.resume:  # optionally resume from a checkpoint
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
            print("searching dir")
            path_resume = os.path.join(p2, get_last_checkpoint(p2))
            # print('path exists, last checkpoint found: {}'.format(path_resume))
        if path_resume is not None:
            print("=> loading checkpoint {}".format(path_resume))
            if args.PS:
                model = HardNetPS()
                checkpoint = torch.load(path_resume)
                model.load_state_dict(checkpoint)
                model.cuda()
            else:
                checkpoint = torch.load(path_resume)
                try:
                    args.start_epoch = checkpoint["epoch"]
                    model.load_state_dict(checkpoint["state_dict"])
                except:
                    print("loading subset of weights")
                    pom = model.state_dict()
                    pom.update(checkpoint)
                    model.load_state_dict(pom)
                try:
                    optimizer1.load_state_dict(checkpoint["optimizer"])
                except:
                    print("optimizer not loaded")
        else:
            print("=> no checkpoint found")
            # print('=> no checkpoint found at {}'.format(path_resume))

    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        train(train_loader, model, optimizer1, epoch, False, WBSLoader=None)
        for test_loader in test_loaders:
            test(test_loader["dataloader"], model, epoch, test_loader["name"], args)


if __name__ == "__main__":
    tst = get_test_loaders()
    DSs = []
    DSs += [One_DS(Args_Brown("Datasets/liberty.pt", 2, True, default_transform), group_id=[0])]
    # DSs += [One_DS(Args_Brown('Datasets/liberty_harris.pt', 2, True, default_transform), group_id=[1])]
    # DSs += [One_DS(Args_Brown('Datasets/notredame.pt', 2, True, default_transform), group_id=[2])]
    # DSs += [One_DS(Args_Brown('Datasets/notredame_harris.pt', 2, True, default_transform), group_id=[3])]
    # DSs += [One_DS(Args_Brown('Datasets/yosemite.pt', 2, True, default_transform), group_id=[4])]
    # DSs += [One_DS(Args_Brown('Datasets/yosemite_harris.pt', 2, True, default_transform), group_id=[5])]
    # DSs += [One_DS(Args_Brown('Datasets/hpatches_split_view_train.pt', 2, True, default_transform), group_id=list(range(6,12)))]
    # DSs += [One_DS(Args_AMOS('Datasets/AMOS_views_v3/Train', 1, split_name, args.n_patch_sets, get_WF_from_string(args.weight_function), True, transform_AMOS,
    #                         args.patch_gen, args.cams_in_batch, masks_dir='Datasets/AMOS_views_v3/Masks'), group_id=list(range(12)))]

    # group_id determines sampling scheme - one group_id is chosen randomly for each batch, single dataset may be in more group_id
    # then the relative_batch_size (any positive number - applies as a ratio) determines how many patches are chosen from each dataset for inidividual batch
    # each batch has size args.batch_size (may differ slightly if args.batch_size is not divisible by sum of relative sizes)
    wrapper = DS_wrapper(DSs, args.n_triplets, args.batch_size)
    os.makedirs(os.path.join(args.model_dir, save_name), exist_ok=True)
    with open(os.path.join(args.model_dir, save_name, "setup.txt"), "w") as f:
        for d in wrapper.datasets:
            print(d.__dict__, file=f)

    print("----------------\nsplit_name: {}".format(split_name))
    print("save_name: {}".format(save_name))
    model = HardNet().cuda()
    main(wrapper, tst, model)
    print("Train end, saved: {}".format(save_name))
    # send_email(recipient='milan.pultar@gmail.com', ignore_host='milan-XPS-15-9560') # useful fo training, change the recipient address for yours or comment this out

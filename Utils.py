import torch, cv2, random, copy, os, argparse, sys, PIL, gc, pickle, sklearn
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as dset
from torch.autograd import Variable
from copy import deepcopy
from EvalMetrics import ErrorRateAt95Recall, AP, prec_recall_curve#, ErrorRateFDRAt95Recall, convertFDR2FPR, convertFPR2FDR
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization
from tqdm import tqdm
from glob import glob
from fire import Fire
from PIL import Image
from enum import Enum
import torch.utils.data as data
from WBSDataset import WBSDataset
# from HandCraftedModules import HessianResp

# resize image to size 32x32
cv2_scale = lambda x: cv2.resize( x, dsize=(32, 32), interpolation=cv2.INTER_LINEAR )
# reshape image
np_reshape32 = lambda x: np.reshape(x, (32, 32, 1))
np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))

def zeros_like(x):
    assert x.__class__.__name__.find('Variable') != -1 or x.__class__.__name__.find('Tensor') != -1, "Object is neither a Tensor nor a Variable"
    y = torch.zeros(x.size())
    if x.is_cuda:
       y = y.cuda()
    if x.__class__.__name__ == 'Variable':
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find('Tensor') != -1:
        return torch.zeros(y)

def ones_like(x):
    assert x.__class__.__name__.find('Variable') != -1 or x.__class__.__name__.find('Tensor') != -1, "Object is neither a Tensor nor a Variable"
    y = torch.ones(x.size())
    if x.is_cuda:
       y = y.cuda()
    if x.__class__.__name__ == 'Variable':
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find('Tensor') != -1:
        return torch.ones(y)

def batched_forward(model, data, batch_size, **kwargs):
    n_patches = len(data)
    if n_patches > batch_size:
        bs = batch_size
        n_batches = n_patches / bs + 1
        for batch_idx in range(n_batches):
            st = batch_idx * bs
            if batch_idx == n_batches - 1:
                if (batch_idx + 1) * bs > n_patches:
                    end = n_patches
                else:
                    end = (batch_idx + 1) * bs
            else:
                end = (batch_idx + 1) * bs
            if st >= end:
                continue
            if batch_idx == 0:
                first_batch_out = model(data[st:end], kwargs)
                out_size = torch.Size([n_patches] + list(first_batch_out.size()[1:]))
                #out_size[0] = n_patches
                out = torch.zeros(out_size);
                if data.is_cuda:
                    out = out.cuda()
                out = Variable(out)
                out[st:end] = first_batch_out
            else:
                out[st:end,:,:] = model(data[st:end], kwargs)
        return out
    else:
        return model(data, kwargs)

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def CircularGaussKernel(kernlen=None, circ_zeros = False, sigma = None, norm = True):
    assert ((kernlen is not None) or sigma is not None)
    if kernlen is None:
        kernlen = int(2.0 * 3.0 * sigma + 1.0)
        if (kernlen % 2 == 0):
            kernlen = kernlen + 1
        # halfSize = kernlen / 2
    halfSize = kernlen / 2
    r2 = float(halfSize*halfSize)
    if sigma is None:
        sigma2 = 0.9 * r2
        # sigma = np.sqrt(sigma2)
    else:
        sigma2 = 2.0 * sigma * sigma
    x = np.linspace(-halfSize,halfSize,kernlen)
    xv, yv = np.meshgrid(x, x, sparse=False, indexing='xy')
    distsq = (xv)**2 + (yv)**2
    kernel = np.exp(-( distsq/ (sigma2)))
    if circ_zeros:
        kernel *= (distsq <= r2).astype(np.float32)
    if norm:
        kernel /= np.sum(kernel)
    return kernel

def generate_2dgrid(h,w, centered = True):
    if centered:
        x = torch.linspace(-w/2+1, w/2, w)
        y = torch.linspace(-h/2+1, h/2, h)
    else:
        x = torch.linspace(0, w-1, w)
        y = torch.linspace(0, h-1, h)
    grid2d = torch.stack([y.repeat(w,1).t().contiguous().view(-1), x.repeat(h)],1)
    return grid2d

def generate_3dgrid(d, h, w, centered = True):
    if type(d) is not list:
        if centered:
            z = torch.linspace(-d/2+1, d/2, d)
        else:
            z = torch.linspace(0, d-1, d)
        dl = d
    else:
        z = torch.FloatTensor(d)
        dl = len(d)
    grid2d = generate_2dgrid(h,w, centered = centered)
    grid3d = torch.cat([z.repeat(w*h,1).t().contiguous().view(-1,1), grid2d.repeat(dl,1)],dim = 1)
    return grid3d

def zero_response_at_border(x, b):
    if (b < x.size(3)) and (b < x.size(2)):
        x[:, :,  0:b, :] =  0
        x[:, :,  x.size(2) - b: , :] =  0
        x[:, :, :,  0:b] =  0
        x[:, :, :,   x.size(3) - b: ] =  0
    else:
        return x * 0
    return x

class GaussianBlur(nn.Module):
    def __init__(self, sigma=1.6):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        return
    def calculate_weights(self,  sigma):
        kernel = CircularGaussKernel(sigma = sigma, circ_zeros = False)
        h,w = kernel.shape
        halfSize = float(h) / 2.;
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1,1,h,w);
    def forward(self, x):
        w = Variable(self.buf)
        if x.is_cuda:
            w = w.cuda()
        return F.conv2d(F.pad(x, (self.pad,self.pad,self.pad,self.pad), 'replicate'), w, padding = 0)

def batch_eig2x2(A):
    trace = A[:,0,0] + A[:,1,1]
    delta1 = (trace*trace - 4 * ( A[:,0,0]*  A[:,1,1] -  A[:,1,0]* A[:,0,1]))
    mask = delta1 > 0
    delta = torch.sqrt(torch.abs(delta1))
    l1 = mask.float() * (trace + delta) / 2.0 +  1000.  * (1.0 - mask.float())
    l2 = mask.float() * (trace - delta) / 2.0 +  0.0001  * (1.0 - mask.float())
    return l1,l2

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    return

def adjust_learning_rate(optimizer, orig_lr, batch_size, n_triplets, epochs):
# Updates the learning rate given the learning rate decay.
# The routine has been implemented according to the original Lua SGD optimizer
    out = 0
    for group in optimizer.param_groups:
        if 'no_grad' in group.keys():
            continue
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = orig_lr * ( 1.0 - float(group['step']) * float(batch_size) / (n_triplets * float(epochs)) )
        out = group['lr']
    return out

def create_optimizer(hardnet, new_lr, optimizer_name, wd, unet=None):
# setup optimizer
    if optimizer_name == 'sgd': # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
        # optimizer = optim.SGD( model.parameters(), lr=new_lr, momentum=0.9, dampening=0.9, weight_decay=wd )
        if unet is None:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, hardnet.parameters()), lr=new_lr, momentum=0.9, dampening=0.9, weight_decay=wd)
        else:
            optimizer = optim.SGD( [{'params':hardnet.parameters(), 'lr':0, 'no_grad':True}, {'params':unet.parameters()}], lr=new_lr, momentum=0.9, dampening=0.9, weight_decay=wd)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(hardnet.parameters(), lr=new_lr, weight_decay=wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(optimizer_name))
    return optimizer

class HardNet(nn.Module):
# HardNet model definition
    def __init__(self, grayscale=True):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False) if grayscale else nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        self.features.apply(HardNet.weights_init)
    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        input = input.float()
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass

class HardNetPS(nn.Module):
    def __init__(self):
        super(HardNetPS, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias=True)
        )

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class TripletPhotoTour(dset.PhotoTour):
# From the PhotoTour Dataset it generates triplet samples
# note: a triplet is composed by a pair of matching images and one of different class.

    mean = {'notredame': 0.4854, 'yosemite': 0.4844, 'liberty': 0.4437, 'hpatches_dataset': 0.4210, 'amos': 0.0, 'amos_10K': 0.0, 'AMOS_another_split_10K': 0.0,
            'notredame_harris': 0.4854, 'yosemite_harris': 0.4844, 'liberty_harris': 0.4437}
    std = {'notredame': 0.1864, 'yosemite': 0.1818, 'liberty': 0.2019, 'hpatches_dataset': 0.2245, 'amos': 0.0, 'amos_10K': 0.0, 'AMOS_another_split_10K': 0.0,
           'notredame_harris': 0.1864, 'yosemite_harris': 0.1818, 'liberty_harris': 0.2019}
    lens = {'notredame': 468159, 'yosemite': 633587, 'liberty': 450092, 'hpatches_dataset': 1659944, 'amos': 0.0, 'amos_10K': 0.0, 'AMOS_another_split_10K': 0.0,
            'liberty_harris': 379587, 'yosemite_harris': 450912, 'notredame_harris': 325295}

    def __init__(self, train=True, transform=None, n_triplets = 1000, batch_size=None, load_random_triplets=False, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes - 1)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes - 1)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes - 1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes - 1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            n3 = np.random.randint(0, len(indices[c2]) - 1)
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img)
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
        if args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0, 2, 1)
                img_p = img_p.permute(0, 2, 1)
                if self.out_triplets:
                    img_n = img_n.permute(0, 2, 1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:, :, ::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:, :, ::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:, :, ::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)


def test(test_loader, model, epoch, logger_test_name, args):
    model.eval()

    labels, distances = [], []

    mean_losses = []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        data_a, data_p, label = Variable(data_a.cuda(), volatile=True), Variable(data_p.cuda(), volatile=True), Variable(label)
        # print(len(data_a))
        # print(data_a.shape)
        out_a, out_p = model(data_a), model(data_p)

        if args.batch_reduce == 'L2Net':
            loss = loss_L2Net( out_a, out_p, anchor_swap=args.anchorswap, margin=args.margin, loss_type=args.loss )
        else:
            loss = loss_HardNet( out_a, out_p, margin=args.margin, anchor_swap=args.anchorswap, anchor_ave=False, batch_reduce=args.batch_reduce, loss_type=args.loss )
            loss = loss.mean()
            mean_losses += [np.mean(loss.data.cpu().numpy())]

        dists = torch.sqrt( torch.sum((out_a - out_p) ** 2, 1) )  # euclidean distance
        distances.append( dists.data.cpu().numpy().reshape(-1, 1) )
        # ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append( label.data.cpu().numpy().reshape(-1, 1) )

        if batch_idx % args.log_interval == 0:
            pbar.set_description(logger_test_name + ' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)) )

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))

    mean_loss = np.mean(np.array(mean_losses))

    print('mean loss:{}'.format(mean_loss))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\33[0m'.format(fpr95))
    print('\33[91mTest set: Accuracy(AP): {:.8f}\33[0m'.format(AP(labels, distances)))

    if args.test:
        curve = prec_recall_curve(labels, distances)
        curve_path = 'prec_recall_curves/curve_'+logger_test_name+'.pickle'
        pickle.dump(curve, open(curve_path, 'wb'))
        print('saved '+curve_path)

        # p_out = 'PR_curve_intra_{}.pickle'.format(t)
        # pickle.dump(PR_curve, open(p_out, 'wb'))
        # print('saved {}'.format(p_out))
        auc = sklearn.metrics.auc(curve[1][::-1], curve[0][::-1])
        print('auc = {}'.format(auc))
        print('')

    # if (args.enable_logging):
    #     logger.log_value(logger_test_name + ' fpr95', fpr95)

class Interface():
    def symlinks(self, src, dst):
        os.makedirs(dst, exist_ok=True)
        for f in glob(os.path.join(src, '*')):
            os.symlink(os.path.abspath(f), f.replace(src, dst))

    def hom_in_HP(self, dir_in, dir_out):
        for d in tqdm(glob(os.path.join(dir_in, '*'))):
            fs = [os.path.join(d, '{}.ppm'.format(c)) for c in range(1,7)]
            imgs = [cv2.imread(f) for f in fs]
            homs = [os.path.join(d, 'H_1_{}'.format(c)) for c in range(2,7)]
            Hs = [' '.join(open(h, 'r').readlines()).strip().replace('\n', ' ').split() for h in homs]
            for i in range(len(Hs)):
                Hs[i] = [float(c) for c in Hs[i]]
                Hs[i] = np.array(Hs[i]).reshape([3,3])
            warps = [cv2.warpPerspective(im, np.linalg.inv(h), imgs[0].shape[:2][::-1]) for im,h in zip(imgs[1:], Hs)]
            do = d.replace(dir_in, dir_out)
            os.makedirs(do, exist_ok=True)

            cv2.imwrite(fs[0].replace(dir_in, dir_out), imgs[0]) # firt has diff size
            for i,f in enumerate(fs[1:]):
                cv2.imwrite(f.replace(dir_in, dir_out), warps[i])

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    BLACK = '\033[1;30m'
    RED = '\033[1;31m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[1;34m'
    PURPLE = '\033[1;35m'
    CYAN = '\033[1;36m'
    WHITE = '\033[1;37m'

    @staticmethod
    def p(color, text):
        print(color + text + bcolors.END)

class TotalDatasetsLoader(data.Dataset):
    def __init__(self, dataset_path, train=True, batch_size=None, n_tuples=5000000, fliprot=False, transform_dict=None, group_id=[0], *arg, **kw):
        super(TotalDatasetsLoader, self).__init__()
        dataset = torch.load(dataset_path)
        data, labels = dataset[0], dataset[1]
        try:
            self.all_txts = dataset[-1] # 'e1, e2, ...'
            if type(self.all_txts)!=type(['some_text']):
                raise Exception()
        except:
            self.all_txts = [''] * len(labels)

        self.split_idxs = [-1] + [torch.max(labels).item()]

        intervals = [(self.split_idxs[i], self.split_idxs[i+1]) for i in range(len(self.split_idxs)-1)]
        self.range_intervals = [list(range(c[0]+1, c[1]+1)) for c in intervals]

        self.data, self.labels = data, labels
        self.transform_dict = transform_dict
        self.train = train
        self.n_tuples = n_tuples
        self.batch_size = batch_size
        self.fliprot = fliprot
        self.triplets = []
        self.group_id = group_id
        if self.train:
            self.generate_triplets(self.labels, self.n_tuples, self.batch_size, intervals=intervals)

    def generate_triplets(self, labels, n_triplets, batch_size, intervals=None):
        def create_indices():
            inds = dict()
            for idx, ind in enumerate(labels):
                ind = ind.item()
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        self.indices = create_indices()

    def __getitem__(self, idx):
        def transform_img(img, transformation=None):
            return transformation(img) if transformation is not None else img

        if len(self.triplets)==0:
            cs = np.random.choice(self.range_intervals[0], size=self.batch_size)
            for c1 in cs:
                if len(self.indices[c1]) == 2:
                    n1, n2 = 0, 1
                else:
                    n1, n2 = np.random.choice(range(len(self.indices[c1])), size=2, replace=False)
                self.triplets += [[self.indices[c1][n1], self.indices[c1][n2], -1]]

        t = self.triplets[0]
        self.triplets = self.triplets[1:]
        a, p = self.data[t[0]], self.data[t[1]] # t[2] would be negative, not used

        img_a = transform_img(a, self.transform_dict[self.all_txts[t[0]]] if self.all_txts[t[0]] in self.transform_dict.keys() else self.transform_dict['default'])
        img_p = transform_img(p, self.transform_dict[self.all_txts[t[1]]] if self.all_txts[t[1]] in self.transform_dict.keys() else self.transform_dict['default'])

        if self.fliprot: # transform images if required
            if random.random() > 0.5: # do rotation
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
            if random.random() > 0.5: # do flip
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))

        return img_a, img_p

    def __len__(self):
        return 9999999999

class FORMAT(Enum):
    AMOS = 0
    Brown = 1

class Args_Brown():
    def __init__(self, path:str, relative_batch_size:int, fliprot:bool, transform_dict:dict):
        self.path = path
        self.batch_size = relative_batch_size
        self.fliprot = fliprot
        self.transform_dict = transform_dict

class Args_AMOS():
    def __init__(self, tower_dataset, split_name, n_patch_sets, weight_function, batch_size, fliprot, transform, patch_gen, cams_in_batch):
        self.tower_dataset = tower_dataset
        self.split_name = split_name
        self.n_patch_sets = n_patch_sets
        self.weight_function = weight_function
        self.batch_size = batch_size
        self.fliprot = fliprot
        self.transform = transform
        self.patch_gen = patch_gen
        self.cams_in_batch = cams_in_batch

class One_DS():
    def __init__(self, args, group_id:[int]=[0]):
        if isinstance(args, Args_Brown):
            self.__dict__ = args.__dict__.copy()
            self.format = FORMAT.Brown
        elif isinstance(args, Args_AMOS):
            self.__dict__ = args.__dict__.copy()
            self.format = FORMAT.AMOS
        else:
            raise('incorrect args class')
        self.group_id = group_id
        # bcolors.p(bcolors.YELLOW, str(self.__dict__))

class DS_wrapper():
    def prepare_epoch(self):
        kwargs = {'num_workers': 1, 'pin_memory': True}
        loaders = []
        for c in self.datasets:
            if c.format == FORMAT.Brown:
                loaders += [TotalDatasetsLoader(train=True, load_random_triplets=False, batch_size=c.batch_size, dataset_path=os.path.abspath(c.path), fliprot=c.fliprot,
                                                n_tuples=self.n_tuples, transform_dict=c.transform_dict, group_id=c.group_id)]
            elif c.format == FORMAT.AMOS:
                loaders += [WBSDataset(root=c.tower_dataset, split_name=c.split_name, n_patch_sets=c.n_patch_sets,
                                       weight_function=c.weight_function, grayscale=True, download=False, group_id=c.group_id,
                                       n_tuples=self.n_tuples, batch_size=c.batch_size, fliprot=c.fliprot, transform=c.transform, cams_in_batch=c.cams_in_batch, patch_gen=c.patch_gen)]
            else:
                raise('invalid DS format')

        self.gid_to_loaders = {}
        for gid in self.group_ids:
            rel_loaders = [c for c in loaders if gid in c.group_id]
            sum_of_sizes = sum([c.batch_size for c in rel_loaders])
            for loader in rel_loaders:
                loader.pom_batch_size = int((loader.batch_size / sum_of_sizes) * self.batch_size)
                # print(loader.pom_batch_size, loader.batch_size, (loader.batch_size / sum_of_sizes))
                loader.n_tuples = int(loader.n_tuples * (loader.pom_batch_size / self.batch_size))
            self.gid_to_loaders[gid] = [iter(torch.utils.data.DataLoader(c, batch_size=c.pom_batch_size, shuffle=False, **kwargs)) for c in rel_loaders]

        for loader in loaders:
            # print(loader)
            bcolors.p(bcolors.YELLOW, str(loader.__dict__))

    def __init__(self, datasets:[One_DS], n_tuples, batch_size, fliprot=False):
        self.n_tuples = n_tuples
        self.group_ids = list(set().union(*[c.group_id for c in datasets]))
        self.gid_to_DS = {}
        self.b_size = batch_size
        self.datasets = datasets
        self.fliprot = fliprot
        self.batch_size = batch_size
        self.prepare_epoch()

    def __getitem__(self, idx):
        if idx > int(self.n_tuples / self.batch_size):
            raise StopIteration
        gid = random.choice(self.group_ids)
        data = next(self.gid_to_loaders[gid][0])
        data_a, data_p = data
        for loader in self.gid_to_loaders[gid][1:]:
            pom_a, pom_p = next(loader)
            data_a = torch.cat((data_a.float(), pom_a.float()))
            data_p = torch.cat((data_p.float(), pom_p.float()))
        return data_a, data_p

    def __len__(self):
        return int(self.n_tuples / self.batch_size)

if __name__ == '__main__':
    Fire(Interface)

    # python Utils.py symlinks ../Process_DS/Datasets/Train ../Process_DS/Datasets_illum/Train
    # python hom-in-HP /home/milan/Prace/CMP/hpatches-sequences-release ../HP_hom
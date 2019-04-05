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
    for group in optimizer.param_groups:
        if 'no_grad' in group.keys():
            continue
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = orig_lr * ( 1.0 - float(group['step']) * float(batch_size) / (n_triplets * float(epochs)) )
    return

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
                img = self.transform(img.numpy())
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
    #fdr95 = ErrorRateFDRAt95Recall(labels, 1.0 / (distances + 1e-8))

    #fpr2 = convertFDR2FPR(fdr95, 0.95, 50000, 50000)
    #fpr2fdr = convertFPR2FDR(fpr2, 0.95, 50000, 50000)

    mean_loss = np.mean(np.array(mean_losses))

    print('mean loss:{}'.format(mean_loss))
    #print('\33[91mTest set: Accuracy(FDR95): {:.8f}\n\33[0m'.format(fdr95))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\33[0m'.format(fpr95))
    print('\33[91mTest set: Accuracy(AP): {:.8f}\33[0m'.format(AP(labels, distances)))
    #print('\33[91mTest set: Accuracy(FDR2FPR): {:.8f}\n\33[0m'.format(fpr2))
    #print('\33[91mTest set: Accuracy(FPR2FDR): {:.8f}\n\33[0m'.format(fpr2fdr))

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

    #fpr2 = convertFDR2FPR(round(fdr95,2), 0.95, 50000, 50000)
    #fpr2fdr = convertFPR2FDR(round(fpr2,2), 0.95, 50000, 50000)

    #print('\33[91mTest set: Accuracy(FDR2FPR): {:.8f}\n\33[0m'.format(fpr2))
    #print('\33[91mTest set: Accuracy(FPR2FDR): {:.8f}\n\33[0m'.format(fpr2fdr))

    if (args.enable_logging):
        logger.log_value(logger_test_name + ' fpr95', fpr95)

class Interface():
    def symlinks(self, src, dst):
        os.makedirs(dst, exist_ok=True)
        for f in glob(os.path.join(src, '*')):
            # print(f.replace(src, dst))
            # exit(0)
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

if __name__ == '__main__':
    Fire(Interface)

    # python Utils.py symlinks ../Process_DS/Datasets/Train ../Process_DS/Datasets_illum/Train
    # python hom-in-HP /home/milan/Prace/CMP/hpatches-sequences-release ../HP_hom
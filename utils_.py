# import math, numpy as np, sys, PIL, random, os, smtplib, socket, shutil, subprocess, cv2
import math, PIL, random, shutil, cv2
import torch.nn.init
# import torchvision.transforms as transforms
# from email.mime.text import MIMEText
from PIL import Image
from fire import Fire
from glob import glob
from os.path import join as pjoin
from os.path import dirname as getdir
from os.path import basename as getbase
from os.path import splitext
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from read_write_model import *
import time


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

class GaussianBlur(torch.nn.Module):
    def __init__(self, sigma=1.6):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        return
    def calculate_weights(self,  sigma):
        kernel = CircularGaussKernel(sigma = sigma, circ_zeros = False)
        h,w = kernel.shape
        halfSize = float(h) / 2.
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1,1,h,w)
    def forward(self, x):
        w = self.buf
        if x.is_cuda:
            w = w.cuda()
        return F.conv2d(F.pad(x, [self.pad,self.pad,self.pad,self.pad], 'replicate'), w, padding = 0)

def batched_forward(model, data, batch_size, **kwargs):
    n_patches = len(data)
    if n_patches > batch_size:
        bs = batch_size
        n_batches = int(n_patches / bs + 1)
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
                # first_batch_out = model(data[st:end], kwargs)
                first_batch_out = model(data[st:end])
                out_size = torch.Size([n_patches] + list(first_batch_out.size()[1:]))
                # out_size[0] = n_patches
                out = torch.zeros(out_size)
                if data.is_cuda:
                    out = out.cuda()
                out[st:end] = first_batch_out
            else:
                # out[st:end, :, :] = model(data[st:end], kwargs)
                out[st:end, :, :] = model(data[st:end])
        return out
    else:
        return model(data, kwargs)

def generate_2dgrid(h, w, centered=True):
    if centered:
        x = torch.linspace(-w / 2 + 1, w / 2, w)
        y = torch.linspace(-h / 2 + 1, h / 2, h)
    else:
        x = torch.linspace(0, w - 1, w)
        y = torch.linspace(0, h - 1, h)
    grid2d = torch.stack([y.repeat(w, 1).t().contiguous().view(-1), x.repeat(h)], 1)
    return grid2d

def generate_3dgrid(d, h, w, centered=True):
    if type(d) is not list:
        if centered:
            z = torch.linspace(-d / 2 + 1, d / 2, d)
        else:
            z = torch.linspace(0, d - 1, d)
        dl = d
    else:
        z = torch.FloatTensor(d)
        dl = len(d)
    grid2d = generate_2dgrid(h, w, centered=centered)
    grid3d = torch.cat([z.repeat(w * h, 1).t().contiguous().view(-1, 1), grid2d.repeat(dl, 1)], dim=1)
    return grid3d

def zero_response_at_border(x, b):
    if (b < x.size(3)) and (b < x.size(2)):
        x[:, :, 0:b, :] = 0
        x[:, :, x.size(2) - b :, :] = 0
        x[:, :, :, 0:b] = 0
        x[:, :, :, x.size(3) - b :] = 0
    else:
        return x * 0
    return x

def batch_eig2x2(A):
    trace = A[:, 0, 0] + A[:, 1, 1]
    delta1 = trace * trace - 4 * (A[:, 0, 0] * A[:, 1, 1] - A[:, 1, 0] * A[:, 0, 1])
    mask = delta1 > 0
    delta = torch.sqrt(torch.abs(delta1))
    l1 = mask.float() * (trace + delta) / 2.0 + 1000.0 * (1.0 - mask.float())
    l2 = mask.float() * (trace - delta) / 2.0 + 0.0001 * (1.0 - mask.float())
    return l1, l2

def normal_df(x, mu=0, sigma=0.01):
    left = 1.0 / torch.sqrt(2.0 * math.pi * sigma*sigma)
    right = torch.exp(-(x-mu)*(x-mu) / (2 * sigma*sigma))
    return left * right

def get_rotation_matrix(angles_in_radians):
    sin_a = np.sin(angles_in_radians)
    cos_a = np.cos(angles_in_radians)
    return np.stack([np.stack([cos_a, sin_a], 1),
                     np.stack([-sin_a, cos_a], 1)], axis=2)

def rectifyAffineTransformationUpIsUpNP(A, eps=1e-10):
    det = np.sqrt(np.abs(A[:,0,0]*A[:,1,1] - A[:,0,1]*A[:,1,0]) + eps)
    b2a2 = np.sqrt(A[:,0,0]**2 + A[:,0,1]**2 + eps)
    aux = (A[:,0,1]*A[:,1,1] + A[:,0,0]*A[:,1,0])
    return np.stack([np.stack([b2a2 / det, aux / (b2a2 * det)], 1),
                     np.stack([np.zeros(det.shape), det / b2a2], 1)], axis=2)

def get_good_sets(info, count_thr=3, loss_thr=sys.maxsize): # returns indices to patch_sets
    losses = info['losses']
    counts = info['counts']

    summed_losses = np.sum(losses, 1)
    summed_counts = np.sum(counts, 1)

    all_sampled = np.min(counts, 1)
    a = (all_sampled >= count_thr)
    b = (summed_losses / summed_counts) < loss_thr
    mask = a * b
    idxs = np.arange(mask.shape[0])

    ### DELETE
    # summed_losses = np.sum(losses, 1)
    # summed_losses /= summed_counts
    # c = (summed_losses > 1.1) + (summed_losses <= 0.9)
    # mask = mask * c
    ### DELETE
    return idxs[mask]

def get_patches_loss(info): # returns losses, has two dims
    losses = info['losses']
    counts = info['counts']
    losses[counts>0] /= counts[counts>0]

    ### DELETE
    # losses[losses > 1.1] = 1.7
    # losses[losses <= 0.9] = 0.3
    ### DELETE
    return losses

# def send_email(recipient='milan.pultar@gmail.com', ignore_host='milan-XPS-15-9560', text=''): # you can use for longer training
#     msg = MIMEText(text)
#
#     if socket.gethostname() == ignore_host:
#         return
#     msg["Subject"] = socket.gethostname() + " just finished running a job "# + os.path.basename(__main__.__file__)
#     msg["From"] = "clustersgpu@gmail.com"
#     msg["To"] = recipient
#
#     s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
#     s.ehlo()
#     s.login("clustersgpu@gmail.com", "4c46bc24732")
#     s.sendmail("clustersgpu@gmail.com", recipient, msg.as_string())
#     s.quit()

class printc:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[1;30m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    PURPLE = "\033[1;35m"
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"

    @staticmethod
    def blue(*text):
        printc.uni(printc.BLUE, text)
    @staticmethod
    def green(*text):
        printc.uni(printc.GREEN, text)
    @staticmethod
    def yellow(*text):
        printc.uni(printc.YELLOW, text)
    @staticmethod
    def red(*text):
        printc.uni(printc.RED, text)
    @staticmethod
    def uni(color, text:tuple):
        print(color + ' '.join([str(x) for x in text]) + printc.END)

def get_laf_scale(LAF: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    out = LAF[..., 0:1, 0:1] * LAF[..., 1:2, 1:2] - LAF[..., 1:2, 0:1] * LAF[..., 0:1, 1:2] + eps
    return out.abs().sqrt()

def lookslikeimage(f):
    exts = ['.ppm', '.jpg', '.jpeg', '.png', '.bmp']
    return sum([f.lower().endswith(c) for c in exts]) > 0

def become_deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dict_add(dictionary:dict, key, value, acc='list'):
    if key not in dictionary.keys():
        if acc=='list':
            dictionary[key] = []
        elif acc=='set':
            dictionary[key] = set()
        else:
            assert False, 'only list or set'
    dictionary[key] += [value]

class measure_time():
    def __init__(self):
        pass
    def __enter__(self):
        self.start_time = time.time()
    def __exit__(self, type, value, traceback):
        print('time elapsed', time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time)))

class Interface:
    def resize_all(self,
                   # dir_in='Datasets/AMOS-views/AMOS-test-1',
                   dir_in='/home/milan/School/2019-2020/B4M33TDV/HWs/Inputs/Room',
                   dir_out='/home/milan/School/2019-2020/B4M33TDV/HWs/Inputs-resized/Room',
                   maxsize=(1000,1000),
                   mdepth=sys.maxsize,
                   rotate=False,
                   ): # resizes all images indirectory recursively
        for (dirpath, dirnames, filenames) in os.walk(dir_in):
            d = len([c for c in dirpath.replace(dir_in, '').split('/') if c!=''])
            if d>mdepth:
                continue
            for f in filenames:
                if not lookslikeimage(f):
                    continue
                in_path = os.path.join(dirpath,f)
                out_path = in_path.replace(dir_in, dir_out)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                print(out_path)
                img = Image.open(in_path)
                if rotate: # for photos taken vertically
                    img = img.rotate(-90, expand=1)
                img.thumbnail(maxsize, Image.ANTIALIAS)
                img.save(out_path)

    def upscale_all(self,
                    dir_in='/home/milan/School/2019-2020/B4M33TDV/HWs/Inputs',
                    dir_out='/home/milan/School/2019-2020/B4M33TDV/HWs/Inputs-resized',
                    # dir_in='Datasets/AMOS-views/AMOS-views-v4',
                    # dir_out='Datasets/AMOS-views/AMOS-views-v4-upscaled',
                    scale=2,
                    mdepth=sys.maxsize,
                    rotate=False,
                    ):
        for (dirpath, dirnames, filenames) in os.walk(dir_in):
            d = len([c for c in dirpath.replace(dir_in, '').split('/') if c != ''])
            if d > mdepth:
                continue
            for f in sorted(filenames):
                if not lookslikeimage(f):
                    continue
                in_path = os.path.join(dirpath, f)
                out_path = in_path.replace(dir_in, dir_out)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                print(out_path)
                img = Image.open(in_path)
                if rotate: # for photos taken vertically
                    img = img.rotate(-90, expand=1)
                img = img.resize([c*scale for c in (img.width,img.height)], PIL.Image.BILINEAR)
                img.save(out_path)

    def one_img_per_folder(self,
                   dir_in='Datasets/AMOS-views/AMOS-test-1-downsized',
                   dir_out='Datasets/AMOS-views/AMOS-test-1-downsized-split',
                   ):
        for p in glob(os.path.join(dir_in, '*')):
            if not os.path.isdir(p):
                continue
            print(p)
            for f in glob(os.path.join(p, '*')):
                p_out = p+'-'+os.path.splitext(os.path.basename(f))[0]
                p_out = os.path.join(p_out, os.path.basename(f))
                p_out = p_out.replace(dir_in, dir_out)
                os.makedirs(os.path.dirname(p_out), exist_ok=True)
                shutil.copyfile(f, p_out)

    def get_depths(self, dir_in='Datasets/AMOS-views/AMOS-views-v4-upscaled', dir_mega='../MegaDepth/demo.py'):
        dir_in = os.path.relpath(dir_in, os.path.dirname(dir_mega))
        dir_out = os.path.join(os.path.dirname(dir_in), os.path.basename(dir_in)+'-depths')
        print('dir_out', dir_out)

        dirs = glob(os.path.join(dir_in, '*'))
        dirs = [c for c in dirs if os.path.isdir(c)]
        os.chdir(os.path.dirname(dir_mega))

        os.makedirs(dir_out, exist_ok=True)
        for d in dirs:
            paths = glob(os.path.join(d, '*'))
            for pin in paths:
                pout = pin.replace(dir_in, dir_out)
                os.makedirs(os.path.dirname(pout), exist_ok=True)
                os.system(' '.join(['python demo.py --p_in',pin,'--p_out',pout]))

    def transPS(self, dir_in='/home/pultami1/PS-Dataset/PS-Dataset'):
        # transform PS-DS to liberty format
        dout = pjoin(getdir(dir_in), getbase(dir_in)+'_trans')
        print(dout)
        folders = glob(pjoin(dir_in, '*'))
        os.makedirs(dout, exist_ok=True)
        for f in tqdm(folders):
            info = open(pjoin(f, 'patch_info.txt'), 'r').readlines()
            ids = [int(c.split(',')[0]) for c in ''.join(info).strip().split()]
            ids = torch.tensor(ids)
            patches = torch.load(pjoin(f, 'patchImg.bin'))
            patches = patches.squeeze(1)

            pout = pjoin(dout, 'PS-'+getbase(f)+'.pt')
            torch.save((patches, ids), pout)

    def extract_patches(self,
                        dir_in='Datasets/Phototourism/trevi_fountain/dense',
                        dir_out='Datasets/Phototourism',
                        ):
        printc.yellow('\n'.join(['Input arguments:'] + [str(x) for x in sorted(locals().items()) if x[0] != 'self']))
        cams, imgs, pts = read_model(path=pjoin(dir_in,'sparse'), ext='.bin')
        print('found points', len(pts))
        paths = glob(pjoin(dir_in, 'images', '*'))
        images = {}
        for p in tqdm(paths):
            images[getbase(p)] = np.asarray(PIL.Image.open(p))
        ids3D = []
        patches = []
        ids_cam = []
        for i in tqdm(list(pts.keys())):
            image_ids = pts[i].image_ids
            point2D_idxs = pts[i].point2D_idxs
            patches_one = []
            ids3D_one = []
            ids_cam_one = []
            for a,b in zip(image_ids,point2D_idxs):
                img_data = imgs[a]
                D2pt = img_data.xys[b]
                img = PIL.Image.fromarray(images[img_data.name], 'RGB').convert('L')
                w, h = img.size[0], img.size[1]
                left, top, right, bottom = D2pt[0] - 32, D2pt[1] - 32, D2pt[0] + 32, D2pt[1] + 32
                if not (left>0 and top>0 and right<w-1 and bottom<h-1): # no black rectangles
                    continue
                patch = img.crop((left, top, right, bottom))
                patch = torch.tensor(np.asarray(patch))
                ids3D_one += [i]
                ids_cam_one += [a]
                patches_one += [patch]
            if len(patches_one) > 1:
                patches += patches_one
                ids3D += ids3D_one
                ids_cam += ids_cam_one
        print('stacking')
        patches = torch.stack(patches, 0)
        ids3D = torch.tensor(ids3D)
        ids_cam = torch.tensor(ids_cam)
        save_path = pjoin(dir_out, getbase(getdir(dir_in))+'.pt')
        print('saving to', save_path)
        torch.save({'patches':patches, 'labels':ids3D, 'cam_ids':ids_cam}, save_path)

    def extract_patches_rgb(self,
                        dir_in='Datasets/Phototourism/colosseum_exterior/dense',
                        dir_out='Datasets/Phototourism',
                        which = 'labelscolo.pt',
                        ):
        printc.yellow('\n'.join(['Input arguments:'] + [str(x) for x in sorted(locals().items()) if x[0] != 'self']))
        cams, imgs, pts = read_model(path=pjoin(dir_in,'sparse'), ext='.bin')
        print('found points', len(pts))
        paths = glob(pjoin(dir_in, 'images', '*'))
        images = {}
        for p in tqdm(paths):
            images[getbase(p)] = np.asarray(PIL.Image.open(p))
        ids3D = []
        patches = []
        ids_cam = []

        subset = torch.load(which)
        for i in tqdm(list(pts.keys())):
            if i not in subset:
                continue
            image_ids = pts[i].image_ids
            point2D_idxs = pts[i].point2D_idxs
            patches_one = []
            ids3D_one = []
            ids_cam_one = []
            for a,b in zip(image_ids,point2D_idxs):
                img_data = imgs[a]
                D2pt = img_data.xys[b]
                img = PIL.Image.fromarray(images[img_data.name], 'RGB')
                w, h = img.size[0], img.size[1]
                left, top, right, bottom = D2pt[0] - 32, D2pt[1] - 32, D2pt[0] + 32, D2pt[1] + 32
                if not (left>0 and top>0 and right<w-1 and bottom<h-1): # no black rectangles
                    continue
                patch = img.crop((left, top, right, bottom))
                # patch = torch.tensor(np.asarray(patch))
                patch = torch.as_tensor(np.asarray(patch))
                ids3D_one += [i]
                ids_cam_one += [a]
                patches_one += [patch]
            if len(patches_one) > 1:
                patches += patches_one
                ids3D += ids3D_one
                ids_cam += ids_cam_one
        print('stacking')
        patches = torch.stack(patches, 0)
        ids3D = torch.tensor(ids3D)
        ids_cam = torch.tensor(ids_cam)
        save_path = pjoin(dir_out, getbase(getdir(dir_in))+'_RGB.pt')
        print('saving to', save_path)
        torch.save({'patches':patches, 'labels':ids3D, 'cam_ids':ids_cam}, save_path)

    def filter_sets(self,
                    # path_ds='Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_maxsets:2000_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:AMOS-masks.pt',
                    path_ds='Datasets/Phototourism/hagia_sophia_interior.pt',
                    # path_stats='Models/id:0_arch:h7_ds:v4_loss:tripletMargin_mpos:1.0_mneg:1.0_lr:0.0_maxsets:2000_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:AMOS-masks_tps:5000000_CamsB:5_resume_ep:1_bs:3072_pos:2/stats_0.npy',
                    path_stats='Models/id:0_arch:h7_ds:hagia_sophia_interior_loss:tripletMargin_mpos:1.0_mneg:1.0_lr:0.0_maxsets:2000_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:AMOS-masks_tps:10000000_CamsB:5_resume_ep:1_bs:3072_pos:2/stats_0.npy',
                    # path_stats='Models/id:0_arch:h7_ds:liberty_loss:tripletMargin_mpos:1.0_mneg:1.0_lr:0.0_maxsets:2000_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:AMOS-masks_tps:10000000_CamsB:5_resume_ep:1_bs:3072_pos:2/stats_0.npy',
                    # path_stats='Models/id:0_arch:h7_ds:brandenburg_gate_loss:tripletMargin_mpos:1.0_mneg:1.0_lr:0.0_maxsets:2000_sigmas-v:e011_thr:0.00016_WF:Hessian_PG:new_masks:AMOS-masks_tps:10000000_CamsB:5_resume_ep:1_bs:3072_pos:2/stats_0.npy',
                    fraction = 0.5,
                    higher = False,
                    middle = False,
                    ):
        printc.yellow('\n'.join(['Input arguments:'] + [str(x) for x in sorted(locals().items()) if x[0] != 'self']))
        ds = torch.load(path_ds)
        stats = np.load(path_stats, allow_pickle=True).item()
        e = stats.get('edges_sets')
        c = stats.get('counts_sets')
        if type(ds) == type({}): # AMOS
            if middle:
                raise NotImplemented()
            idxs = np.argsort(e / c)
            if higher:
                idxs = idxs[::-1].copy()
            idxs = idxs[:int(fraction * len(idxs))]
            ds['patch_sets'] = ds['patch_sets'][idxs]
            ds['LAFs'] = ds['LAFs'][idxs]
            ds['cam_idxs'] = ds['cam_idxs'][idxs]
            if 'collisions' in ds.keys():
                ds['collisions'] = [ds['collisions'][i] for i in idxs]
        else: # liberty
            all_set_ids = np.sort(torch.unique(ds[1]).data.cpu().numpy())
            print('found sets:',len(all_set_ids))
            c = c[all_set_ids]
            e = e[all_set_ids]
            mean_e = e/c # priority
            if middle:
                mean_e[np.isnan(mean_e)] = -999 # do not pick nans
                aux_idxs = np.logical_and((mean_e > -0.1), (mean_e < 0.1))
                print(aux_idxs)
            else:
                if higher:
                    mean_e[np.isnan(mean_e)] = -999 # do not pick nans
                    aux_idxs = np.argsort(mean_e)[::-1].copy()
                else:
                    mean_e[np.isnan(mean_e)] = 999
                    aux_idxs = np.argsort(mean_e).copy()
                aux_idxs = aux_idxs[:int(fraction * len(aux_idxs))] # idxs to all_set_ids
            set_ids = set(all_set_ids[aux_idxs]) # picked set_ids
            print('found patches:',len(ds[1]))
            print('selected sets:',len(set_ids))
            idxs = [i for i,c in enumerate(ds[1].data.cpu().numpy()) if c in set_ids] # idxs to data
            print('selected patches:',len(idxs))
            ds = (ds[0][idxs],ds[1][idxs])
        if middle:
            p_out = splitext(path_ds)[0]+'_middleedges.pt'
        else:
            p_out = splitext(path_ds)[0]+'_fraction:'+str(fraction)+'_higher:'+str(int(higher))+'.pt'
        print('saving to', p_out)
        torch.save(ds, open(p_out, 'wb'))

    def extract_hps(self, # extract hpatches
                         dir_in='../hpatches-release',
                         dir_out='Datasets/HPatches',
                         # splits=('illum', 'view'),  # only illum, view available
                         splits=['illum'],  # only illum, view available
                         types = ("e1", "e2", "e3", "e4", "e5", "ref", "h1", "h2", "h3", "h4", "h5", "t1", "t2", "t3", "t4", "t5"),
                         suffix='all',
                         # types = ("e1", "e2", "e3", "e4", "e5"),
                         # suffix='easy',
                         # types=("h1", "h2", "h3", "h4", "h5"),
                         # suffix='hard',
                         # types=("t1", "t2", "t3", "t4", "t5"),
                         # suffix='tough',
                         exclude=set(),
                         ):
        printc.yellow('\n'.join(['Input arguments:'] + [str(x) for x in sorted(locals().items()) if x[0] != 'self']))
        save_path = pjoin(dir_out, '_'.join(["HPs", "-".join(splits), suffix + ".pt"]))
        print("save_path:", save_path)

        print("splits:", splits)
        patches, labels, offset = [], [], 0
        txts = []
        hpatches_sequences = [x[1] for x in os.walk(dir_in)][0]
        pbar = tqdm(hpatches_sequences, total=len(hpatches_sequences))
        for dir in pbar:
            pbar.set_description(dir)
            name = getbase(dir)
            if sum([c[0] + "_" in name for c in splits]) == 0: # checks for i_, v_
                continue
            if name in exclude:
                print("XXXXX", name)
                continue
            for type in types:
                sequence_path = pjoin(dir_in, dir, type) + ".png"
                image = cv2.imread(sequence_path, 0)
                h, w = image.shape
                n_patches = int(h / w)
                for i in range(n_patches):
                    patch = image[i * (w): (i + 1) * (w), 0:w]
                    patch = np.array(cv2.resize(patch, (64, 64)), dtype=np.uint8)
                    patches += [patch]
                    labels += [offset + i]
                    txts += [type]
            offset += n_patches
        patches = torch.ByteTensor(np.array(patches, dtype=np.uint8))
        labels = torch.LongTensor(labels)
        print('patches.shape:', patches.shape)
        res = (patches, labels, txts)

        os.makedirs(dir_out, exist_ok=True)
        print("saving to ", save_path)
        torch.save(res, open(save_path, "wb"))


if __name__ == "__main__":
    Fire(Interface)
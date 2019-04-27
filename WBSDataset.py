import os, sys, random, glob, errno, json, torch, math, gc, torchvision, multiprocessing, cv2, itertools
from os import path
import numpy as np
from PIL import Image
from pprint import pprint
import torch.utils.data as data
from joblib import Parallel, delayed
from utils import download_url, check_integrity, send_email
import pathos.pools as pp
import torchvision.transforms as tforms
from copy import deepcopy
from tqdm import tqdm
from random import randint
import torch.nn as nn


def get_img_set_num(dir_name):
    img_set = os.listdir(dir_name)
    return len(img_set)


def LAF2A(laf, h, w, PS=32):
    min_size = float(min(h, w))
    A = deepcopy(laf)
    A[0, :2] = 2.0 * laf[0, :2] / float(PS)
    A[1, :2] = 2.0 * laf[1, :2] / float(PS)
    s = float(PS) * np.sqrt(np.abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]))
    A[0, 2] = A[0, 2] - s / 2.0
    A[1, 2] = A[1, 2] - s / 2.0
    return A


def rotate(l, n):
    return l[n:] + l[:n]


class NMS2d(nn.Module):
    def __init__(self, kernel_size=17, threshold=0):  # kernel_size = 3
        super(NMS2d, self).__init__()
        self.MP = nn.MaxPool2d(kernel_size, stride=1, return_indices=False, padding=kernel_size / 2)
        self.eps = 1e-10
        self.th = threshold
        return

    def forward(self, x):
        # local_maxima = self.MP(x)
        # if x!=self.MP(x):
        #     return x
        # else:
        #     return torch.Tensor(0)

        if self.th > self.eps:
            return x * (x > self.th).float() * ((x + self.eps - self.MP(x)) > 0).float()
        else:
            # return ((x - self.MP(x) + self.eps) > 0).float() * x
            return (x == self.MP(x)).float() * x


### example of correct weight function
# from HandCraftedModules import HessianResp
# HR = HessianResp()
# def weightFHR(pilimg):
#    with torch.no_grad():
#        return HessianResp()((tforms.ToTensor()(pilimg)).unsqueeze(0).mean(dim=1, keepdim = True), 1.0).squeeze().numpy()
######
class WBSDataset(data.Dataset):
    """`HPatches: A benchmark and evaluation of handcrafted and learned local descriptors <https://hpatches.github.io/>`_ Dataset.
    Args:
        root (string): Root directory where directories with image sets are.
        path_to_train_test_split(string or None): path to json text file with train-test split. If None, random is generated
        split_name (string): Name of the train-test split to load.
        dataset_name(string):
        n_patch_sets(int): Number of correspondences to generate
        n_positives(int, default: 2): Number of positive examples to generate, if more than 2 images exist
        max_tilt(float, default 6.0): Maximum anisotropic scaling factor, when generate patch, same for all patches in set.
        patch_size(int, default 128): Output patch size
        grayscale(bool, default True): Output grayscale patches
        weight_function(callable, optional): function for weigthing random sampling. E.g. some saliency
        Takes PIL image, outputs numpy array of the same w,h
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version.
        download (bool, optional): If true, down: loads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root,
        path_to_train_test_split=None,
        split_name="full",
        transform=None,
        n_patch_sets=1000,
        n_positives=2,
        patch_size=96,
        weight_function=None,
        grayscale=True,
        max_tilt=1.0,
        group_id=[0],
        border=5,
        download=False,
        overwrite=False,
        n_tuples=2000,
        batch_size=1024,
        fliprot=False,
        patch_gen="oneImg",
        masks_dir=None,
        mark_patches_dir=None,
        new_angles=0,
        cams_in_batch=0,
        spx=0,
        spy=0,
        save_losses=False,
        NMS=False,
    ):
        self.tpl = tforms.ToPILImage()
        self.ttsr = tforms.ToTensor()
        self.img_ext = set(["jpg", "jpeg", "png", "ppm", "bmp"])
        self.root = os.path.expanduser(root)
        self.path_to_train_test_split = path_to_train_test_split
        self.split_name = split_name
        self.transform = transform
        self.n_patch_sets = n_patch_sets
        self.n_positives = n_positives
        self.patch_size = patch_size
        self.weight_function = weight_function
        self.grayscale = grayscale
        self.max_tilt = max_tilt
        self.download = download
        self.border = border
        self.batch_size = batch_size
        self.n_tuples = n_tuples
        self.fliprot = fliprot
        self.group_id = group_id

        self.NMS = NMS

        self.see_all_in_imtower = False  # True is deprecated
        self.patch_gen = patch_gen
        self.masks_dir = masks_dir
        self.masks_global = True  # False does not work atm
        self.mark_patches_dir = mark_patches_dir
        self.new_angles = new_angles
        self.cams_in_batch = cams_in_batch

        self.spx = spx
        self.spy = spy
        self.save_losses = save_losses

        self.cam_idx = 0

        # if self.train:
        self.data_file = os.path.join(self.root, "_".join([str(c) for c in [path.basename(root).lower(), self.split_name, "train.pt"]]))
        # else:
        #     self.data_file = os.path.join(self.root, path.basename(root).lower() + '_' + self.split_name + '_test.pt')

        if self.path_to_train_test_split is None:
            self.img_fnames = None
        else:
            with open(self.path_to_train_test_split) as splits_file:
                data = json.load(splits_file)
            if self.train:
                if "train" in data[self.split_name]:
                    self.img_fnames = set(data[self.split_name]["train"])
                else:
                    print("No train in selected split, use test")
                    self.img_fnames = set(data[self.split_name]["test"])
            else:
                self.img_fnames = set(data[self.split_name]["test"])
        self.process_dataset(overwrite=overwrite)
        # if download:
        #    self.download()
        if not self._check_datafile_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")
        self.pairs = []
        if self.patch_gen == "watchGood":
            self.indices = self.generate_pairs_new(n_tuples, batch_size)
        else:
            self.indices = self.generate_tuples(n_tuples, batch_size)

    # def __iter__(self):
    #     pass
    #     return self

    def __getitem__(self, index):
        # def __getitem__(self, index):
        # Args: index (int): Index
        # Returns: list: [data1, data2, ..., dataN]
        #     all_cams = np.unique(self.cam_idxs.numpy().astype(np.int)).tolist()
        #     cam_idxs = [np.array([i for i,x in enumerate(self.cam_idxs) if x==c]) for c in all_cams]
        #     n_batches = int(self.n_pairs / self.batch_size)
        #     cur_allowed_cams = all_cams
        #     for _ in tqdm(range(n_batches)):

        if len(self.pairs) == 0:
            self.cur_allowed_cams = self.get_source_cams(self.all_cams, self.cur_allowed_cams, self.cams_in_batch)
            self.allowed_idxs = list(itertools.chain.from_iterable([self.cam_idxs[c] for c in [self.all_cams.index(x) for x in self.cur_allowed_cams]]))
            cs = np.random.choice(self.allowed_idxs, size=self.batch_size)
            for c1 in cs:
                ns = np.random.choice(list(range(len(self.patch_sets[c1]))), self.n_positives)
                self.pairs += [[c1, *ns]]
            # return torch.LongTensor(np.array(pairs))

        # c1n1n2 = self.indices[

        c1n1n2 = self.pairs[0]
        self.pairs = self.pairs[1:]
        data = self.patch_sets[c1n1n2[0]]
        imgs = [data[c] for c in c1n1n2[1:]]
        if self.transform is not None:
            imgs = [self.transform(c) for c in imgs]
        if self.fliprot:  # transform images if required
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                imgs = [c.permute(0, 2, 1) for c in imgs]
            if do_flip:
                # img_a = torch.from_numpy(deepcopy(img_a.detach().numpy()[:,:,::-1]))
                # img_p = torch.from_numpy(deepcopy(img_p.detach().numpy()[:,:,::-1]))

                imgs = [torch.from_numpy(deepcopy(c.numpy()[:, :, ::-1])) for c in imgs]

        return (imgs, c1n1n2[0]) if self.save_losses else imgs

    def __len__(self):
        # return self.n_pairs
        return 999999999

    def _check_downloaded(self):
        # return os.path.exists(self.data_dir)
        pass

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)

    def get_source_cams(self, available_cameras, prev_cameras, cams_in_batch):
        if cams_in_batch <= 0 or cams_in_batch > len(available_cameras):
            cams_in_batch = len(available_cameras)
        rest_classes = list(set.difference(set(available_cameras), set(prev_cameras)))
        from_rest = min(len(rest_classes), cams_in_batch)
        again_classes = list(set.difference(set(available_cameras), set(rest_classes)))
        from_again = cams_in_batch - from_rest
        out = list(random.sample(rest_classes, from_rest))
        out += list(random.sample(again_classes, from_again))
        return out

    def generate_tuples(self, n_pairs, batch_size):
        pairs = []
        self.all_cams = np.unique(self.cam_idxs.numpy().astype(np.int)).tolist()

        self.cam_idxs = [np.array([i for i, x in enumerate(self.cam_idxs) if x == c]) for c in self.all_cams]
        n_batches = int(n_pairs / batch_size)
        self.cur_allowed_cams = self.all_cams
        # for _ in tqdm(range(n_batches)):
        #     cur_allowed_cams = self.get_source_cams(all_cams, cur_allowed_cams, self.cams_in_batch)
        #     allowed_idxs = list(itertools.chain.from_iterable([cam_idxs[c] for c in [all_cams.index(x) for x in cur_allowed_cams]]))
        #
        #     cs = np.random.choice(allowed_idxs, size=batch_size)
        #
        #     for c1 in cs:
        #         # num_pos = len(self.patch_sets[c1])
        #         ns = np.random.choice( list(range(len(self.patch_sets[c1]))), self.n_positives )
        #         # if num_pos == 2:  # hack to speed up process
        #         #     n1, n2 = 0, 1
        #         # else:
        #         #     n1 = np.random.randint(0, num_pos)
        #         #     n2 = np.random.randint(0, num_pos)
        #         #     while n1 == n2:
        #         #         n2 = np.random.randint(0, num_pos)
        #         # pairs += [[c1, n1, n2]]
        #         pairs += [[c1, *ns]]
        # return torch.LongTensor(np.array(pairs))

    def generate_pairs_new(self, n_pairs, batch_size):
        pairs = []
        n_classes = len(self.patch_sets)
        already_idxs = set()
        for x in tqdm(range(n_pairs)):
            if len(already_idxs) >= batch_size:
                already_idxs = set()
            # if len(already_imgs) >= self.all_img_idxs[-1].numpy()[0]+1:
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:  # or self.all_img_idxs[c1].numpy()[0] in already_imgs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            num_pos = len(self.patch_sets[c1])
            n1 = np.random.randint(0, num_pos)
            while self.idxs_good[c1, n1] == 0:
                n1 = np.random.randint(0, num_pos)
            n2 = np.random.randint(0, num_pos)
            while n1 == n2 or self.idxs_good[c1, n2] == 0:
                n2 = np.random.randint(0, num_pos)
            pairs.append([c1, n1, n2])
        return torch.LongTensor(np.array(pairs))

    def generate_image_tower(self, dir_name):
        fullpath = os.path.join(self.root, dir_name)
        files = os.listdir(fullpath)
        img_fnames = []
        for ext in self.img_ext:
            img_fnames = img_fnames + [x for x in files if x.endswith(ext)]
        img_fnames = sorted(img_fnames)
        trans_fnames = sorted([x for x in files if os.path.splitext(x)[1] == ""])
        images = []
        # registered_images = []
        assert (len(trans_fnames) + 1 == len(img_fnames)) or (len(trans_fnames) == 0)
        for i in range(len(img_fnames)):
            fname = img_fnames[i]
            img = Image.open(os.path.join(fullpath, fname))

            ######### random shift ... this code is deprecated, it influences probability maps only
            # if self.sx is not 0 or self.sy is not 0:
            #     img = np.array(img)
            #     img = img[:, :, ::-1].copy()
            #     rows, cols, _ = img.shape
            #
            #     # sp_x, sp_y = 0.01, 0.01
            #     max_x = int(self.sx * cols)
            #     max_y = int(self.sy * rows)
            #     M = np.float32([[1, 0, randint(0,2*max_x)-max_x], [0, 1, randint(0,2*max_y)-max_y]])
            #     img = cv2.warpAffine(img, M, (cols, rows))
            #
            #     cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     img = Image.fromarray(img)
            # else:
            #     print('no rotation used')
            ######### random shift

            if self.grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            if i != 0:
                if len(trans_fnames) > 0:
                    h_fname = os.path.join(fullpath, trans_fnames[i - 1])
                    H = np.loadtxt(h_fname)
                    H = H / H[2, 2]
                    if np.abs(H - np.eye(3)).max() > 1e-5:  # do transform
                        img = img.transform((img.width, img.height), Image.PERSPECTIVE, H.ravel(), Image.BICUBIC)
                else:
                    assert (img.width == images[0].width) and (img.height == images[0].height)
            images.append(img)
        return images

    def get_rotation_matrix(self, angle_in_radians):
        angle_in_radians = angle_in_radians.reshape(-1, 1, 1)
        sin_a = np.sin(angle_in_radians)
        cos_a = np.cos(angle_in_radians)
        A1_x = np.concatenate([cos_a, -sin_a], axis=2)
        A2_x = np.concatenate([sin_a, cos_a], axis=2)
        transform = np.concatenate([A1_x, A2_x], axis=1)
        return transform

    def rectifyAffineTransformationUpIsUpNP(self, A):
        det = np.sqrt(np.abs(A[:, 0, 0] * A[:, 1, 1] - A[:, 1, 0] * A[:, 0, 1] + 1e-10))
        b2a2 = np.sqrt(A[:, 0, 1] * A[:, 0, 1] + A[:, 0, 0] * A[:, 0, 0])
        A1_ell = np.concatenate([(b2a2 / det).reshape(-1, 1, 1), 0 * det.reshape(-1, 1, 1)], axis=2)
        A2_ell = np.concatenate([((A[:, 1, 1] * A[:, 0, 1] + A[:, 1, 0] * A[:, 0, 0]) / (b2a2 * det)).reshape(-1, 1, 1), (det / b2a2).reshape(-1, 1, 1)], axis=2)
        return np.concatenate([A1_ell, A2_ell], axis=1)

    def get_mask_pyr(self, img, mask):
        img = deepcopy(img)
        mask = np.copy(mask)

        mask_pyr, pyr = [], []
        # img = deepcopy(img)
        pyr.append(img)
        # w1,h1 = w,h

        w1 = img.width
        h1 = img.height

        curr_mask = Image.fromarray(np.uint8(mask * 255))
        while min(w1, h1) > 10:
            w1, h1 = pyr[-1].size
            new = pyr[-1].copy()
            # print new.width,curr_mask.width
            if self.weight_function is not None:
                weighted_mask = self.weight_function(new)
            else:
                weighted_mask = np.ones((h1, w1))

            if self.NMS:
                weighted_mask = NMS2d(kernel_size=3)(torch.autograd.Variable(torch.Tensor(np.expand_dims(weighted_mask, 0)))).data.cpu().numpy().squeeze(0)

            mask_pyr.append(weighted_mask * np.array(curr_mask))
            new.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            curr_mask.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            pyr.append(new)

        return mask_pyr

    def get_pyr_images(self, img, mask):
        img = deepcopy(img)
        mask = np.copy(mask)

        mask_pyr, pyr = [], []
        # img = deepcopy(img)
        pyr.append(img)
        # w1,h1 = w,h

        w1 = img.width
        h1 = img.height

        curr_mask = Image.fromarray(np.uint8(mask * 255))
        while min(w1, h1) > 10:
            w1, h1 = pyr[-1].size
            new = pyr[-1].copy()
            # print new.width,curr_mask.width
            if self.weight_function is not None:
                weighted_mask = self.weight_function(new)
            else:
                weighted_mask = np.ones((h1, w1))
            mask_pyr.append(weighted_mask * np.array(curr_mask))
            new.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            curr_mask.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            pyr.append(new)

        return pyr

    def generateLAFs(self, nLAFs, imtower, border=5, mask=None):
        angles = np.random.uniform(0, np.pi, (nLAFs))
        # angles2 = np.random.uniform(0, np.pi, (nLAFs));
        tilts = np.random.uniform(1.0, self.max_tilt, (nLAFs))
        init_scale = 16
        w = imtower[0].width
        h = imtower[0].height
        scales = np.exp(np.random.uniform(np.log(init_scale), np.log(np.minimum(h, w) / 4.0 - 2 * border), (nLAFs)))
        centers = np.zeros((nLAFs, 2)) - 1
        sc = init_scale
        if mask is None:
            mask = np.ones((h, w))
        mask_pyr = []
        pyr = []
        img = deepcopy(imtower[0])
        pyr.append(img)
        w1, h1 = w, h
        curr_mask = Image.fromarray(np.uint8(mask * 255))
        while min(w1, h1) > 10:
            w1, h1 = pyr[-1].size
            new = pyr[-1].copy()
            # print new.width,curr_mask.width
            if self.weight_function is not None:
                weighted_mask = self.weight_function(new)
            else:
                weighted_mask = np.ones((h1, w1))

            if self.NMS:
                weighted_mask = NMS2d(kernel_size=3)(torch.autograd.Variable(torch.Tensor(np.expand_dims(weighted_mask, 0)))).data.cpu().numpy().squeeze(0)

            mask_pyr.append(weighted_mask * np.array(curr_mask))
            new.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            curr_mask.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            pyr.append(new)
        idx = 0
        w1, h1 = w, h
        sc = init_scale
        while sc <= min(w, h) / 4.0:
            sc_idx = scales >= sc
            n_curr_level = sc_idx.sum()
            if n_curr_level == 0:
                sc = int(sc * 2)
                idx += 1
                continue
            mask1 = mask_pyr[idx]
            h1, w1 = mask1.shape
            if (w1 - 2 * border <= 0) or (h1 - 2 * border <= 0):
                print(n_curr_level, "non fixed")
                break
            mask_curr = mask1[border : h1 - border, border : w1 - border]
            mask_curr /= mask_curr.sum()
            x = np.linspace(sc + border, w - 1 - (sc + border), w1 - 2 * border)
            y = np.linspace(sc + border, h - 1 - (sc + border), h1 - 2 * border)
            xv, yv = np.meshgrid(x, y)
            n_centers_possible = xv.flatten()[:].shape[0]
            idxs = np.arange(0, n_centers_possible)
            # print idxs.shape,mask_curr.flatten().shape
            out_idxs = np.random.choice(idxs, n_curr_level, p=mask_curr.flatten())
            centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1, 1), yv.flatten()[out_idxs].reshape(-1, 1)], axis=1)
            sc = int(sc * 2)
            idx += 1
        rot = self.get_rotation_matrix(angles)
        tilt_matrix = np.repeat(np.eye(2).reshape(1, 2, 2), nLAFs, axis=0)
        tilt_matrix[:, 0, 0] = np.sqrt(tilts)
        tilt_matrix[:, 1, 1] = np.sqrt(1.0 / tilts)
        A = self.rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1, 1, 1)
        LAFs = np.concatenate([scales.reshape(-1, 1, 1) * A, centers.reshape(-1, 2, 1)], axis=2)
        return LAFs, scales

    def generateLAFs_medianImg(self, nLAFs, imtower, border=5, mask=None):
        angles = np.random.uniform(0, np.pi, (nLAFs))
        # angles2 = np.random.uniform(0, np.pi, (nLAFs));
        tilts = np.random.uniform(1.0, self.max_tilt, (nLAFs))
        init_scale = 16
        w = imtower[0].width
        h = imtower[0].height
        scales = np.exp(np.random.uniform(np.log(init_scale), np.log(np.minimum(h, w) / 4.0 - 2 * border), (nLAFs)))
        centers = np.zeros((nLAFs, 2)) - 1
        sc = init_scale
        if mask is None:
            mask = np.ones((h, w))
        mask_pyr = []
        pyr = []
        # img = deepcopy(imtower[0])

        # computing median
        img_np = np.array([np.array(c) for c in imtower])
        img_np = np.median(img_np, 0)
        img = Image.fromarray(img_np)
        # out = img_np.astype(np.uint8)
        # computing median

        pyr.append(img)
        w1, h1 = w, h
        curr_mask = Image.fromarray(np.uint8(mask * 255))
        while min(w1, h1) > 10:
            w1, h1 = pyr[-1].size
            new = pyr[-1].copy()
            # print new.width,curr_mask.width
            if self.weight_function is not None:
                weighted_mask = self.weight_function(new)
            else:
                weighted_mask = np.ones((h1, w1))

            if self.NMS:
                weighted_mask = NMS2d(kernel_size=3)(torch.autograd.Variable(torch.Tensor(np.expand_dims(weighted_mask, 0)))).data.cpu().numpy().squeeze(0)

            mask_pyr.append(weighted_mask * np.array(curr_mask))
            new.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            curr_mask.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            pyr.append(new)
        idx = 0
        w1, h1 = w, h
        sc = init_scale
        while sc <= min(w, h) / 4.0:
            sc_idx = scales >= sc
            n_curr_level = sc_idx.sum()
            if n_curr_level == 0:
                sc = int(sc * 2)
                idx += 1
                continue
            mask1 = mask_pyr[idx]
            h1, w1 = mask1.shape
            if (w1 - 2 * border <= 0) or (h1 - 2 * border <= 0):
                print(n_curr_level, "non fixed")
                break
            mask_curr = mask1[border : h1 - border, border : w1 - border]
            mask_curr /= mask_curr.sum()
            x = np.linspace(sc + border, w - 1 - (sc + border), w1 - 2 * border)
            y = np.linspace(sc + border, h - 1 - (sc + border), h1 - 2 * border)
            xv, yv = np.meshgrid(x, y)
            n_centers_possible = xv.flatten()[:].shape[0]
            idxs = np.arange(0, n_centers_possible)
            # print idxs.shape,mask_curr.flatten().shape
            out_idxs = np.random.choice(idxs, n_curr_level, p=mask_curr.flatten())
            centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1, 1), yv.flatten()[out_idxs].reshape(-1, 1)], axis=1)
            sc = int(sc * 2)
            idx += 1
        rot = self.get_rotation_matrix(angles)
        tilt_matrix = np.repeat(np.eye(2).reshape(1, 2, 2), nLAFs, axis=0)
        tilt_matrix[:, 0, 0] = np.sqrt(tilts)
        tilt_matrix[:, 1, 1] = np.sqrt(1.0 / tilts)
        A = self.rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1, 1, 1)
        LAFs = np.concatenate([scales.reshape(-1, 1, 1) * A, centers.reshape(-1, 2, 1)], axis=2)
        return LAFs, scales

    def generateLAFs_meanImg(self, nLAFs, imtower, border=5, mask=None):
        angles = np.random.uniform(0, np.pi, (nLAFs))
        # angles2 = np.random.uniform(0, np.pi, (nLAFs));
        tilts = np.random.uniform(1.0, self.max_tilt, (nLAFs))
        init_scale = 16
        w = imtower[0].width
        h = imtower[0].height
        scales = np.exp(np.random.uniform(np.log(init_scale), np.log(np.minimum(h, w) / 4.0 - 2 * border), (nLAFs)))
        centers = np.zeros((nLAFs, 2)) - 1
        sc = init_scale
        if mask is None:
            mask = np.ones((h, w))
        mask_pyr = []
        pyr = []
        # img = deepcopy(imtower[0])

        # computing mean
        img_np = np.array([np.array(c) for c in imtower])
        img_np = np.mean(img_np, 0)
        img = Image.fromarray(img_np)
        # out = img_np.astype(np.uint8)
        # computing mean

        pyr.append(img)
        w1, h1 = w, h
        curr_mask = Image.fromarray(np.uint8(mask * 255))
        while min(w1, h1) > 10:
            w1, h1 = pyr[-1].size
            new = pyr[-1].copy()
            # print new.width,curr_mask.width
            if self.weight_function is not None:
                weighted_mask = self.weight_function(new)
            else:
                weighted_mask = np.ones((h1, w1))

            if self.NMS:
                weighted_mask = NMS2d(kernel_size=3)(torch.autograd.Variable(torch.Tensor(np.expand_dims(weighted_mask, 0)))).data.cpu().numpy().squeeze(0)

            mask_pyr.append(weighted_mask * np.array(curr_mask))
            new.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            curr_mask.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
            pyr.append(new)
        idx = 0
        w1, h1 = w, h
        sc = init_scale
        while sc <= min(w, h) / 4.0:
            sc_idx = scales >= sc
            n_curr_level = sc_idx.sum()
            if n_curr_level == 0:
                sc = int(sc * 2)
                idx += 1
                continue
            mask1 = mask_pyr[idx]
            h1, w1 = mask1.shape
            if (w1 - 2 * border <= 0) or (h1 - 2 * border <= 0):
                print(n_curr_level, "non fixed")
                break
            mask_curr = mask1[border : h1 - border, border : w1 - border]
            mask_curr /= mask_curr.sum()
            x = np.linspace(sc + border, w - 1 - (sc + border), w1 - 2 * border)
            y = np.linspace(sc + border, h - 1 - (sc + border), h1 - 2 * border)
            xv, yv = np.meshgrid(x, y)
            n_centers_possible = xv.flatten()[:].shape[0]
            idxs = np.arange(0, n_centers_possible)
            # print idxs.shape,mask_curr.flatten().shape
            out_idxs = np.random.choice(idxs, n_curr_level, p=mask_curr.flatten())
            centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1, 1), yv.flatten()[out_idxs].reshape(-1, 1)], axis=1)
            sc = int(sc * 2)
            idx += 1
        rot = self.get_rotation_matrix(angles)
        tilt_matrix = np.repeat(np.eye(2).reshape(1, 2, 2), nLAFs, axis=0)
        tilt_matrix[:, 0, 0] = np.sqrt(tilts)
        tilt_matrix[:, 1, 1] = np.sqrt(1.0 / tilts)
        A = self.rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1, 1, 1)
        LAFs = np.concatenate([scales.reshape(-1, 1, 1) * A, centers.reshape(-1, 2, 1)], axis=2)
        return LAFs, scales

    def generateLAFs_own(self, nLAFs, imtower, border=5, mask=None):
        angles = np.random.uniform(0, np.pi, (nLAFs))
        tilts = np.random.uniform(1.0, self.max_tilt, (nLAFs))
        init_scale = 16
        w = imtower[0].width
        h = imtower[0].height
        scales = np.exp(np.random.uniform(np.log(init_scale), np.log(np.minimum(h, w) / 4.0 - 2 * border), (nLAFs)))
        centers = np.zeros((nLAFs, 2)) - 1
        if mask is None:
            mask = np.ones((h, w))

        mask_pyr_list = [self.get_mask_pyr(imt, mask) for imt in imtower]

        idxs_good = np.zeros((nLAFs, len(imtower)), dtype=np.int)

        idx = 0
        sc = init_scale
        while sc <= min(w, h) / 4.0:
            sc_idx = scales >= sc
            n_curr_level = sc_idx.sum()
            if n_curr_level == 0:
                sc = int(sc * 2)
                idx += 1
                continue

            for iii, mask1 in enumerate(mask_pyr_list):
                mask1 = mask1[idx]
                h1, w1 = mask1.shape
                if (w1 - 2 * border <= 0) or (h1 - 2 * border <= 0):
                    print(n_curr_level, "non fixed")
                    break
                mask_curr = mask1[border : h1 - border, border : w1 - border]
                if iii == 0:
                    mask_bin = []
                    mask_sum = np.zeros(mask_curr.shape)
                    mask_nums = np.zeros(mask_curr.shape)
                if mask_curr.sum() < 1e-10:
                    mask_bin += [np.zeros(mask_curr.flatten().shape)]
                    mask_nums += np.reshape(mask_bin[-1], mask_curr.shape)
                    continue
                # mask_curr /= mask_curr.sum()
                x = np.linspace(sc + border, w - 1 - (sc + border), w1 - 2 * border)
                y = np.linspace(sc + border, h - 1 - (sc + border), h1 - 2 * border)
                xv, yv = np.meshgrid(x, y)
                # n_centers_possible = xv.flatten()[:].shape[0]
                # idxs = np.arange(0, n_centers_possible)

                mask_sum += mask_curr
                mask_bin += [mask_curr.flatten()]
                # idxs_top = np.argsort(mask_bin[-1])[::-1][:len(mask_bin[-1])/2]
                idxs_top = mask_bin[-1] > 0.05
                # np.where( (mask_bin[-1][idxs_top]) == 0 )
                mask_bin[-1] = np.zeros(mask_bin[-1].shape)
                mask_bin[-1][idxs_top] = 1
                mask_nums += np.reshape(mask_bin[-1], mask_curr.shape)

            # mask_sum[mask_nums<15] = 0
            mask_sum = NMS2d()(torch.autograd.Variable(torch.Tensor(np.expand_dims(mask_sum, 0)))).data.cpu().numpy().squeeze(0)
            # mask_sum /= mask_sum.sum()
            # idxs_sum = np.argsort(mask_sum.flatten())[::-1]
            # mask_dets = np.empty_like(mask_res[-1]).astype(np.int)
            # for mmm in mask_res:
            #     mask_dets += (mmm > 0).astype(np.int)

            # out_idxs = np.random.choice(idxs, n_curr_level, p=mask_curr.flatten())
            # out_idxs = np.random.choice(idxs, n_curr_level, p=mask_sum.flatten())
            out_idxs = np.argsort(mask_sum.flatten())[::-1][:n_curr_level]
            print("last_response: {}".format(mask_sum.flatten()[out_idxs[-1]]))

            centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1, 1), yv.flatten()[out_idxs].reshape(-1, 1)], axis=1)
            idxs_good[sc_idx, :] = np.array([[mask_bin[kkk].flatten()[jjj] for kkk in range(len(mask_bin))] for jjj in out_idxs])
            sc = int(sc * 2)
            idx += 1

        rot = self.get_rotation_matrix(angles)
        tilt_matrix = np.repeat(np.eye(2).reshape(1, 2, 2), nLAFs, axis=0)
        tilt_matrix[:, 0, 0] = np.sqrt(tilts)
        tilt_matrix[:, 1, 1] = np.sqrt(1.0 / tilts)
        A = self.rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1, 1, 1)
        LAFs = np.concatenate([scales.reshape(-1, 1, 1) * A, centers.reshape(-1, 2, 1)], axis=2)
        return LAFs, idxs_good, scales

    def generateLAFs_sumRes(self, nLAFs, imtower, border=5, mask=None):
        angles = np.random.uniform(0, np.pi, (nLAFs))
        tilts = np.random.uniform(1.0, self.max_tilt, (nLAFs))
        init_scale = 16
        w = imtower[0].width
        h = imtower[0].height
        scales = np.exp(np.random.uniform(np.log(init_scale), np.log(np.minimum(h, w) / 4.0 - 2 * border), (nLAFs)))
        centers = np.zeros((nLAFs, 2)) - 1
        if mask is None:
            mask = np.ones((h, w))

        mask_pyr_list = [self.get_mask_pyr(imt, mask) for imt in imtower]

        idxs_good = np.zeros((nLAFs, len(imtower)), dtype=np.int)

        idx = 0
        sc = init_scale
        while sc <= min(w, h) / 4.0:
            sc_idx = scales >= sc
            n_curr_level = sc_idx.sum()
            if n_curr_level == 0:
                sc = int(sc * 2)
                idx += 1
                continue

            for iii, mask1 in enumerate(mask_pyr_list):
                mask1 = mask1[idx]
                h1, w1 = mask1.shape
                if (w1 - 2 * border <= 0) or (h1 - 2 * border <= 0):
                    print(n_curr_level, "non fixed")
                    break
                mask_curr = mask1[border : h1 - border, border : w1 - border]
                if iii == 0:
                    mask_bin = []
                    mask_sum = np.zeros(mask_curr.shape)
                    mask_nums = np.zeros(mask_curr.shape)
                if mask_curr.sum() < 1e-10:
                    mask_bin += [np.zeros(mask_curr.flatten().shape)]
                    mask_nums += np.reshape(mask_bin[-1], mask_curr.shape)
                    continue
                # mask_curr /= mask_curr.sum()
                x = np.linspace(sc + border, w - 1 - (sc + border), w1 - 2 * border)
                y = np.linspace(sc + border, h - 1 - (sc + border), h1 - 2 * border)
                xv, yv = np.meshgrid(x, y)
                # n_centers_possible = xv.flatten()[:].shape[0]
                # idxs = np.arange(0, n_centers_possible)

                mask_sum += mask_curr
                mask_bin += [mask_curr.flatten()]
                # idxs_top = np.argsort(mask_bin[-1])[::-1][:len(mask_bin[-1])/2]
                idxs_top = mask_bin[-1] > 0.05
                # np.where( (mask_bin[-1][idxs_top]) == 0 )
                mask_bin[-1] = np.zeros(mask_bin[-1].shape)
                mask_bin[-1][idxs_top] = 1
                mask_nums += np.reshape(mask_bin[-1], mask_curr.shape)

            # mask_sum[mask_nums<15] = 0
            # mask_sum = NMS2d()(torch.autograd.Variable(torch.Tensor(np.expand_dims(mask_sum, 0))) ).data.cpu().numpy().squeeze(0)
            mask_sum /= mask_sum.sum()
            # idxs_sum = np.argsort(mask_sum.flatten())[::-1]
            # mask_dets = np.empty_like(mask_res[-1]).astype(np.int)
            # for mmm in mask_res:
            #     mask_dets += (mmm > 0).astype(np.int)

            n_centers_possible = xv.flatten()[:].shape[0]
            idxs = np.arange(0, n_centers_possible)
            # out_idxs = np.random.choice(idxs, n_curr_level, p=mask_curr.flatten())
            out_idxs = np.random.choice(idxs, n_curr_level, p=mask_sum.flatten())
            # out_idxs = np.argsort(mask_sum.flatten())[::-1][:n_curr_level]
            print("last_response: {}".format(mask_sum.flatten()[out_idxs[-1]]))

            centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1, 1), yv.flatten()[out_idxs].reshape(-1, 1)], axis=1)
            idxs_good[sc_idx, :] = np.array([[mask_bin[kkk].flatten()[jjj] for kkk in range(len(mask_bin))] for jjj in out_idxs])
            sc = int(sc * 2)
            idx += 1

        rot = self.get_rotation_matrix(angles)
        tilt_matrix = np.repeat(np.eye(2).reshape(1, 2, 2), nLAFs, axis=0)
        tilt_matrix[:, 0, 0] = np.sqrt(tilts)
        tilt_matrix[:, 1, 1] = np.sqrt(1.0 / tilts)
        A = self.rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1, 1, 1)
        LAFs = np.concatenate([scales.reshape(-1, 1, 1) * A, centers.reshape(-1, 2, 1)], axis=2)
        return LAFs, scales

    def generateLAFs_medianRes(self, nLAFs, imtower, border=5, mask=None):
        angles = np.random.uniform(0, np.pi, (nLAFs))
        tilts = np.random.uniform(1.0, self.max_tilt, (nLAFs))
        init_scale = 16
        w = imtower[0].width
        h = imtower[0].height
        scales = np.exp(np.random.uniform(np.log(init_scale), np.log(np.minimum(h, w) / 4.0 - 2 * border), (nLAFs)))
        centers = np.zeros((nLAFs, 2)) - 1
        if mask is None:
            mask = np.ones((h, w))

        mask_pyr_list = [self.get_mask_pyr(imt, mask) for imt in imtower]

        idxs_good = np.zeros((nLAFs, len(imtower)), dtype=np.int)

        idx = 0
        sc = init_scale
        while sc <= min(w, h) / 4.0:
            sc_idx = scales >= sc
            n_curr_level = sc_idx.sum()
            if n_curr_level == 0:
                sc = int(sc * 2)
                idx += 1
                continue

            for iii, mask1 in enumerate(mask_pyr_list):
                mask1 = mask1[idx]
                h1, w1 = mask1.shape
                if (w1 - 2 * border <= 0) or (h1 - 2 * border <= 0):
                    print(n_curr_level, "non fixed")
                    break
                mask_curr = mask1[border : h1 - border, border : w1 - border]
                if iii == 0:
                    mask_bin = []
                    mask_sum = np.zeros(mask_curr.shape)
                    mask_nums = np.zeros(mask_curr.shape)
                    mask_list = []
                if mask_curr.sum() < 1e-10:
                    mask_bin += [np.zeros(mask_curr.flatten().shape)]
                    mask_nums += np.reshape(mask_bin[-1], mask_curr.shape)
                    continue
                # mask_curr /= mask_curr.sum()
                x = np.linspace(sc + border, w - 1 - (sc + border), w1 - 2 * border)
                y = np.linspace(sc + border, h - 1 - (sc + border), h1 - 2 * border)
                xv, yv = np.meshgrid(x, y)
                # n_centers_possible = xv.flatten()[:].shape[0]
                # idxs = np.arange(0, n_centers_possible)

                mask_sum += mask_curr
                mask_list += [mask_curr]
                mask_bin += [mask_curr.flatten()]
                # idxs_top = np.argsort(mask_bin[-1])[::-1][:len(mask_bin[-1])/2]
                idxs_top = mask_bin[-1] > 0.05
                # np.where( (mask_bin[-1][idxs_top]) == 0 )
                mask_bin[-1] = np.zeros(mask_bin[-1].shape)
                mask_bin[-1][idxs_top] = 1
                mask_nums += np.reshape(mask_bin[-1], mask_curr.shape)

            # mask_sum[mask_nums<15] = 0
            # mask_sum = NMS2d()(torch.autograd.Variable(torch.Tensor(np.expand_dims(mask_sum, 0))) ).data.cpu().numpy().squeeze(0)
            mask_sum = np.median(np.array(mask_list), 0)
            mask_sum /= mask_sum.sum()
            # idxs_sum = np.argsort(mask_sum.flatten())[::-1]
            # mask_dets = np.empty_like(mask_res[-1]).astype(np.int)
            # for mmm in mask_res:
            #     mask_dets += (mmm > 0).astype(np.int)

            n_centers_possible = xv.flatten()[:].shape[0]
            idxs = np.arange(0, n_centers_possible)
            # out_idxs = np.random.choice(idxs, n_curr_level, p=mask_curr.flatten())
            out_idxs = np.random.choice(idxs, n_curr_level, p=mask_sum.flatten())
            # out_idxs = np.argsort(mask_sum.flatten())[::-1][:n_curr_level]
            print("last_response: {}".format(mask_sum.flatten()[out_idxs[-1]]))

            centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1, 1), yv.flatten()[out_idxs].reshape(-1, 1)], axis=1)
            idxs_good[sc_idx, :] = np.array([[mask_bin[kkk].flatten()[jjj] for kkk in range(len(mask_bin))] for jjj in out_idxs])
            sc = int(sc * 2)
            idx += 1

        rot = self.get_rotation_matrix(angles)
        tilt_matrix = np.repeat(np.eye(2).reshape(1, 2, 2), nLAFs, axis=0)
        tilt_matrix[:, 0, 0] = np.sqrt(tilts)
        tilt_matrix[:, 1, 1] = np.sqrt(1.0 / tilts)
        A = self.rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1, 1, 1)
        LAFs = np.concatenate([scales.reshape(-1, 1, 1) * A, centers.reshape(-1, 2, 1)], axis=2)
        return LAFs, scales

    def draw_centers(self, img, centers, dir_name, dir_out, scales):
        # pil_image = Image.open('Image.jpg').convert('RGB')
        open_cv_image = np.array(img.convert("RGB"))
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        for ct, sc in zip(centers, scales):
            open_cv_image = cv2.circle(open_cv_image, (int(ct[0]), int(ct[1])), 20, (0, 0, 255))
        cv2.imwrite(os.path.join(dir_out, dir_name + ".png"), open_cv_image)
        return

    def generate_LAFs_and_patches_from_dir(self, dir_name, nLAFs):
        imtower = self.generate_image_tower(dir_name)
        w = imtower[0].width
        h = imtower[0].height
        PS = self.patch_size

        if self.masks_dir is not None:
            if self.masks_global:
                mask = Image.open(os.path.join(self.masks_dir, dir_name.split("_")[0]) + ".png")
                mask = np.array(mask) > 0
                print("cam mask found")
            else:
                fullpath = os.path.join(self.root, dir_name)
                files = os.listdir(fullpath)
                img_fnames = [x for x in files]
                img_fnames = sorted(img_fnames)
        else:
            mask = np.ones((h, w))

        # crop bottom and up 5%:
        # crop_pixs = h / 20
        # print(mask[range(crop_pixs),:].shape)
        # mask[range(crop_pixs), :] = 0
        # mask[range(h-crop_pixs, h), :] = 0

        if self.see_all_in_imtower:
            assert False, "this code is deprecated"
            LAFs_per_img = max(1, int(float(nLAFs) / len(imtower)))
            LAFs, idxs_good = gen_fce(LAFs_per_img, imtower, border=self.border, mask=mask)
            for iii in list(range(len(imtower)))[1:]:
                imtower = rotate(imtower, iii)
                LAFs, idxs_good = np.concatenate((gen_fce(LAFs_per_img, imtower, border=self.border, mask=mask), LAFs), axis=0)
        else:
            if self.patch_gen == "watchGood":
                LAFs, idxs_good, scales = self.generateLAFs_own(nLAFs, imtower, border=self.border, mask=mask)
            elif self.patch_gen == "oneRes":
                LAFs, scales = self.generateLAFs(nLAFs, imtower, border=self.border, mask=mask)
            elif self.patch_gen == "medianImg":
                LAFs, scales = self.generateLAFs_medianImg(nLAFs, imtower, border=self.border, mask=mask)
            elif self.patch_gen == "meanImg":
                LAFs, scales = self.generateLAFs_meanImg(nLAFs, imtower, border=self.border, mask=mask)
            elif self.patch_gen == "sumRes":
                LAFs, scales = self.generateLAFs_sumRes(nLAFs, imtower, border=self.border, mask=mask)
            elif self.patch_gen == "medianRes":
                LAFs, scales = self.generateLAFs_medianRes(nLAFs, imtower, border=self.border, mask=mask)

        centers = LAFs[:, :, 2]
        if self.mark_patches_dir is not None:
            self.draw_centers(imtower[0], centers, dir_name, self.mark_patches_dir, scales)

        patch_tower = []
        tot = tforms.ToTensor()
        angle_range = (-np.pi, np.pi)
        if self.new_angles == 1:
            angle_range = ((self.cam_idx) * (np.pi / 12), (self.cam_idx + 1) * (np.pi / 12))
        print("angle range: {}".format(angle_range))

        angles = np.degrees(np.random.uniform(angle_range[0], angle_range[1], (nLAFs) if (not self.see_all_in_imtower) else (nLAFs * len(imtower))))

        for img in imtower:

            def resamplePatch(LAF, ang, border, spx, spy):
                big_PS = int(1.5 * PS)
                A = LAF2A(LAF, w, h, big_PS)
                left = (big_PS - PS) / 2
                top = (big_PS - PS) / 2
                right = (PS + big_PS) / 2
                bottom = (PS + big_PS) / 2
                # angles2 = np.random.uniform(0, np.pi, (nLAFs))

                ######### random shift ... aplies to each patch individually based on its scale
                if spx is not 0 or spy is not 0:
                    # init_scale = 16
                    # min_scale = 2.0* np.exp(np.log(init_scale)) / float(big_PS)
                    # max_scale = 2.0* np.exp(np.log(np.minimum(h, w) / 4.0 - 2 * border)) / float(big_PS)
                    # shift_coef = (A[0,0] - min_scale) / (max_scale - min_scale)
                    # sx = int(2* spx * w * shift_coef)
                    # sy = int(2* spy * h * shift_coef)
                    sx = int(2 * spx * A[0, 0])
                    sy = int(2 * spy * A[0, 0])  # A[0,0] is the scale
                    A[0, 2] += random.randint(0, sx) - (sx / 2)
                    A[1, 2] += random.randint(0, sy) - (sy / 2)
                ######### random shift

                return img.transform((big_PS, big_PS), Image.AFFINE, A[:2, :].ravel(), Image.BICUBIC).rotate(ang).crop((left, top, right, bottom))

            num_cores = multiprocessing.cpu_count()
            p = pp.ProcessPool(min(8, num_cores))
            patches = p.map(resamplePatch, LAFs, angles, [self.border] * len(angles), [self.spx] * len(angles), [self.spy] * len(angles))
            patch_tower.append(torch.cat([(255 * tot(x).unsqueeze(0)).byte() for x in patches]))
        aaa = torch.cat([pt.unsqueeze(1) for pt in patch_tower], dim=1)

        del patch_tower
        return aaa, torch.from_numpy(LAFs).float(), idxs_good if self.patch_gen == "watchGood" else None

    def process_dataset(self, overwrite=False):
        if self._check_datafile_exists() and not overwrite:
            print("# Found cached data {}".format(self.data_file))
            if self.patch_gen == "watchGood":
                self.patch_sets, self.all_lafs, self.cam_idxs, self.idxs_good = torch.load(self.data_file)
            else:
                self.patch_sets, self.all_lafs, self.cam_idxs = torch.load(self.data_file)
            return

        print("# Not found: {}".format(self.data_file))

        img_dirs = sorted([x for x in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, x))])

        pixels = 0
        for img_dir in img_dirs:
            img_PIL = Image.open(glob.glob(path.join(path.join(self.root, img_dir), "*"))[0]).convert("RGB")
            pixels += img_PIL.size[0] * img_PIL.size[1]
        print("pixels total: {}".format(pixels))

        self.patch_sets = []
        self.all_lafs = []
        self.cam_idxs = []
        self.idxs_good = []
        idx = 0

        def print_log(tag, text):
            print(tag + ": " + text)

        min_num_pics = sys.maxsize
        for i, img_dir in enumerate(img_dirs):
            self.cam_idx = i
            if self.img_fnames is not None:
                if img_dir not in self.img_fnames:
                    continue
            print("cam {}, dir {}".format(i, img_dir))

            img_PIL = Image.open(glob.glob(path.join(path.join(self.root, img_dir), "*"))[0]).convert("RGB")
            num_patches = int(self.n_patch_sets * (float(img_PIL.size[0] * img_PIL.size[1]) / pixels))
            print_log(img_dir, "num_patches={}".format(num_patches))
            patches, lafs, idxs_good = self.generate_LAFs_and_patches_from_dir(img_dir, num_patches)

            min_num_pics = min(min_num_pics, patches.shape[1])
            # print_log(img_dir, 'this_seq_len: '+str(patches.shape[1])+', min_seq_len: '+str(min_num_pics))

            self.idxs_good += [idxs_good]

            self.patch_sets.append(patches)
            self.all_lafs.append(lafs)
            self.cam_idxs.append(torch.ones(int(num_patches), 1) * idx)
            idx += 1

        # self.patch_sets = [ppp[:,:min_num_pics,:,:] for ppp in self.patch_sets]
        print_log("", "# cams: " + str(len(self.patch_sets)))

        self.patch_sets = torch.cat(self.patch_sets, dim=0)
        self.all_lafs = torch.cat(self.all_lafs, dim=0)
        self.cam_idxs = torch.cat(self.cam_idxs, dim=0)
        if self.patch_gen == "watchGood":
            self.idxs_good = np.concatenate(self.idxs_good)
        with open(self.data_file, "wb") as f:
            if self.patch_gen == "watchGood":
                torch.save((self.patch_sets, self.all_lafs, self.cam_idxs, self.idxs_good), f)
            # elif self.patch_gen in ['oneImg', 'sumImg', 'median']:
            else:
                torch.save((self.patch_sets, self.all_lafs, self.cam_idxs), f)

        print("Dataset path: {}".format(self.data_file))
        print("Dataset saved, due to safety, please run again the same command to start training")
        send_email(recipient="milan.pultar@gmail.com", ignore_host="milan-XPS-15-9560")  # useful fo training, change the recipient address for yours or comment this out
        exit(0)


# ds = WBSDataset(root='/home/old-ufo/Dropbox/snavely', n_patch_sets = 10000, grayscale = True, max_tilt = 4.0,
#               weight_function = weightFHR, overwrite = True, border = 10)
# ds = WBSDataset(root='/home/old-ufo/Dropbox/EVD_mik', n_patch_sets = 10000, grayscale = True, max_tilt = 4.0 )

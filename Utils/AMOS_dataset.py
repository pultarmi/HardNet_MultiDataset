import itertools
import torch.utils.data as data
import pathos.pools as pp
import torchvision.transforms as tforms
from copy import deepcopy
from Learning.LAF import LAF2A
from Learning.handcrafted_modules import *
from tqdm import tqdm
from scipy import spatial
from shapely.geometry import Polygon
from Utils.SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from dataclasses import dataclass


@dataclass
class AMOS_dataset(data.Dataset):
    root_dir:str = None  # root (string): Root directory where directories with image sets are.
    data_name:str = 'full'  # split_name (string): Name of the train-test split to load.
    transform:object = None  # transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version.
    Npatch_sets:int = 30000  # n_patch_sets(int): Number of correspondences to generate
    Npositives:int = 2  # n_positives(int, default: 2): Number of positive examples to generate, if more than 2 images exist
    patch_size:int = 96  # patch_size(int, default 128): Output patch size
    weight_function:object = None  # weight_function(callable): function for weigthing random sampling. E.g. some saliency
    max_tilt:float = 1.0  # max_tilt(float, default 6.0): Maximum anisotropic scaling factor, when generate patch, same for all patches in set.
    border:int = 5
    overwrite:bool = False
    Ntuples:int = 2000
    fliprot:bool = True
    patch_gen:str = 'oneImg'
    masks_dir:str = None
    mark_patches_dir:str = None
    cams_in_batch:int = 0
    spx:int = 0
    spy:int = 0
    label_offset:int = 0
    separ_batches:bool = False
    data_path:str = None
    good_pr:float = 0.1
    pairs_imgs:bool = False
    min_sets_per_img:int = -1
    max_sets_per_img:int = sys.maxsize
    good_patches:object = None
    new_batches:bool = False
    use_patchmask:bool = False
    use_collisions:bool = False
    nonmax:bool = False
    sigmas_v:str = 'v1'
    init_thr:float = 0.00001
    scaleit:bool = False
    thrit:bool = False
    to_gauss:bool = False
    gauss_s:float = 30.0
    fixed_MP:bool = False
    depths:str = ''
    AMOS_RGB:bool = False
    only_D:bool = False
    allowed_sets:object = None # np.array

    def __post_init__(self):
        print('new_batches:', self.new_batches)
        print('patch_gen:', self.patch_gen)

        self.tpl = tforms.ToPILImage()
        self.ttsr = tforms.ToTensor()

        if self.root_dir is not None:
            self.root = os.path.expanduser(self.root_dir)
        if self.good_patches is None:
            self.good_patches = []
        if self.data_path is None:
            self.data_path = pjoin(self.root, '_'.join([os.path.basename(self.root_dir), self.data_name+'.pt']))

        self.prepare_dataset(self.Npatch_sets, overwrite=self.overwrite)

        if self.use_collisions:
            printc.green('using collisions')
            self.collisions = self.collisions if hasattr(self, 'collisions') else self.get_collisions()
        else:
            self.collisions = None

    def __getitem__(self, idx):
        t = self.tuples[idx]
        patches = [self.patch_sets[t[0],i] for i in t[1]] # t[0] label, t[1] indices
        if self.transform is not None:
            patches = [self.transform(c) for c in patches]
        if self.fliprot:
            if random.random() > 0.5: # do rot
                patches = [c.permute(0, 2, 1) for c in patches]
            if random.random() > 0.5: # do flip
                patches = [torch.from_numpy(deepcopy(c.numpy()[:, :, ::-1])) for c in patches]
        PSs = torch.tensor(t[0]).repeat(len(patches)).long()
        return {'data':torch.stack(patches, 0), 'PS_idxs':PSs[0],'set_idxs':t[1][0],'labels':PSs+self.label_offset}

    def turn_off(self):
        raise StopIteration
    def __len__(self):
        return len(self.tuples)
    def max_label(self):
        return self.patch_sets.shape[0]-1+self.label_offset
    def min_label(self):
        return 0
    def num_classes(self):
        return self.patch_sets.shape[0]
    def get_labels(self):
        return np.arange(self.patch_sets.shape[0])+self.label_offset

    def generate_tuples(self, bs_seq):
        tuples = []
        all_cams = np.unique(self.cam_idxs.numpy().astype(np.int)).tolist()
        cam_idxs = [np.array([i for i, x in enumerate(self.cam_idxs) if x == c]) for c in all_cams]

        if self.separ_batches:
            good_patches = self.good_patches
            all_idxs = list(itertools.chain.from_iterable([cam_idxs[c] for c in [all_cams.index(x) for x in all_cams]]))
            bad_patches = [x for x in all_idxs if x not in good_patches]
            printc.yellow('good_sets:{}, bad_sets:{}'.format(len(good_patches), len(bad_patches)))
        if self.pairs_imgs:
            self.block_sizes = []

        if self.new_batches:
            print('Generating AMOS tuples (new_batches)')
            def wrap_fce(all_cams, patches_mask, patch_sets_shape, cam_idxs, collisions, n_positives):
                def fce(cur_bsize):
                    unseen_views = all_cams.copy()
                    tuples_cur = []
                    while True:
                        if len(unseen_views) == 0: # repeat
                            unseen_views = all_cams.copy()
                        cur_view = random.choice(unseen_views)
                        unseen_views.remove(cur_view)
                        cur_sidxs = cam_idxs[cur_view]
                        cur_pair = random.sample(list(range(patch_sets_shape)), n_positives)
                        valid_init = np.sum(patches_mask[cur_sidxs][:, np.array(cur_pair)], 1) >= 2  # only detected ones
                        left_sidxs = cur_sidxs[valid_init]

                        chosen = set()
                        for i,cur_sidx in enumerate( np.random.permutation(left_sidxs) ):
                            if collisions is not None and len(set(collisions[cur_sidx]).intersection(chosen)) != 0:
                                continue
                            chosen.add(cur_sidx)
                            tuples_cur += [[cur_sidx, cur_pair]]
                            cur_bsize -= 1
                            if cur_bsize <= 0:
                                break
                        if cur_bsize <= 0:
                            break
                    return tuples_cur
                return fce
            p = pp.ProcessPool(multiprocessing.cpu_count())
            p.restart() # in pathos new instances of pool are somehow tied to one global instance, so restart and close each time
            tuples = p.map(wrap_fce(all_cams.copy(), self.patches_mask, self.patch_sets.shape[1], cam_idxs, self.collisions, self.Npositives), bs_seq)
            p.close() # otherwise script does not exit with threads still running
            tuples = list(itertools.chain.from_iterable(tuples))
        else:
            for batch_size in tqdm(bs_seq, desc='Generating tuples'):
                if self.separ_batches and len(bad_patches) >= batch_size and len(good_patches) >= batch_size:
                    if len(tuples)==0:
                        printc.green('Separ_batches active')
                    if np.random.random() < self.good_pr:
                        cs = random.sample(list(good_patches), batch_size)
                    else:
                        cs = random.sample(list(bad_patches), batch_size)

                    for c1 in cs:
                        ns = np.random.choice(list(range(len(self.patch_sets[c1]))), self.Npositives)
                        tuples += [[c1, ns]]
                else:
                    if self.separ_batches and len(tuples)==0:
                        printc.red('Separ_batches not active')
                    elif self.pairs_imgs:
                        # cur_allowed_cams = self.get_source_cams(all_cams, 1)
                        cur_allowed_cams = get_random_subset(all_cams, 1)
                        allowed_idxs = list(itertools.chain.from_iterable([cam_idxs[c] for c in [all_cams.index(x) for x in cur_allowed_cams]]))

                        set_size = self.patch_sets.shape[1]
                        batch_split = np.array_split(np.arange(batch_size), max(1, 1 + batch_size // len(allowed_idxs)))
                        sizes = [len(c) for c in batch_split]
                        perms = list(itertools.combinations(list(range(set_size)), 2))
                        pairs = random.sample(perms, len(batch_split))

                        self.block_sizes += [sizes]
                        for ns,s in zip(pairs, sizes):
                            cs = random.sample(allowed_idxs, s)
                            for c1 in cs:
                                tuples += [[c1, ns]]
                    else:
                        # cur_allowed_cams = self.get_source_cams(all_cams, self.cams_in_batch)
                        cur_allowed_cams = get_random_subset(all_cams, self.cams_in_batch)
                        other_cams = set(all_cams).difference(set(cur_allowed_cams))

                        allowed_idxs = list(itertools.chain.from_iterable([cam_idxs[c] for c in [all_cams.index(x) for x in cur_allowed_cams]]))
                        other_idxs = list(itertools.chain.from_iterable([cam_idxs[c] for c in [all_cams.index(x) for x in other_cams]]))

                        from_allowed = min(len(allowed_idxs), batch_size)
                        cs = random.sample(allowed_idxs, from_allowed)
                        cs += random.sample(other_idxs, batch_size - from_allowed)
                        for c1 in cs:
                            ns = np.random.choice(np.arange(self.patch_sets.shape[1])[self.patches_mask[c1, :]], self.Npositives)
                            tuples += [[c1, ns]]
        # self.tuples = torch.LongTensor(np.array(tuples)) # because of collate_fn
        self.tuples = tuples

    def generateLAFs_avgImg(self, nLAFs, imgs, border=5, mask=None, init_scale=16, mode='mean'):
        img_np = np.array([np.array(c) for c in imgs])
        if mode in ['median']:
            ref_img = Image.fromarray(np.median(img_np, 0))
        elif mode in ['mean']:
            # print(np.mean(img_np, 0).astype(np.uint8))
            ref_img = Image.fromarray(np.mean(img_np, 0).astype(np.uint8))
        elif mode in ['one']:
            ref_img = Image.fromarray(img_np[0])
        w, h = imgs[0].width, imgs[0].height
        # mask_pyr = self.get_mask_pyr(ref_img, [mask, np.ones((h, w))][mask is None] )
        mask_pyr = get_mask_pyr(ref_img, [mask, np.ones((h, w))][mask is None], self.weight_function, self.scaleit)

        sc_thrs = [2**s for s in range(int(math.log(init_scale,2)), int(math.log(min(w, h) / 4.0, 2))+1 )]
        scales = np.exp(np.random.uniform(np.log(init_scale), np.log(np.minimum(h, w) / 4.0 - 2 * border), nLAFs))
        # scales_cats = np.zeros(nLAFs) - 1
        centers = np.zeros((nLAFs, 2)) - 1
        for mask_cur,sc_thr in zip(mask_pyr,sc_thrs):
            sc_idx = (scales >= sc_thr) * (scales < sc_thr * 2) ### CHANGE
            n_curr_level = sc_idx.sum()
            print('> Scale', sc_thr, 'curr_level', n_curr_level)
            if n_curr_level == 0:
                # print('skipping')
                continue
            h1, w1 = mask_cur.shape
            assert (w1 > 2*border) and (h1 > 2*border), 'non fixed'
            mask_valid = mask_cur[border : h1 - border, border : w1 - border]
            x = np.linspace(sc_thr + border, w - 1 - (sc_thr + border), w1 - 2 * border)
            y = np.linspace(sc_thr + border, h - 1 - (sc_thr + border), h1 - 2 * border)
            xv, yv = np.meshgrid(x, y)
            idxs = np.arange(0, xv.flatten()[:].shape[0])
            if self.nonmax:
                ss = torch.from_numpy(mask_valid).float().cuda()
                # print(ss[0].unsqueeze(0).unsqueeze(0).shape)
                # safe_dist = init_scale / 2
                safe_dist = 50
                MP = torch.nn.MaxPool2d(int(safe_dist)+1-(int(safe_dist)%2), stride=1, padding=int(safe_dist / 2) ).cuda()
                # mm = torch.cat([MP(x.unsqueeze(0).unsqueeze(0)) for x in ss], dim=1).squeeze(0)
                mm = MP(ss.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                # ss[(ss != mm.squeeze(0)) + (ss < thr)] = 0
                # print(torch.sum(ss!=0))
                ss[ss != mm] = 0
                if self.thrit:
                    ss[ss < self.init_thr] = 0
                ss = ss.data.cpu().numpy()
                # print(np.sum(ss!=0))

                out_idxs = np.argsort(ss.ravel())[-n_curr_level:][::-1]
                topN_val = ss.ravel()[out_idxs]
                # print(topN_val)
                # print('zeros:', np.sum(topN_val == 0))
                out_idxs = out_idxs[topN_val != 0]
                aux = np.arange(len(sc_idx))
                aux = aux[sc_idx == 1]
                aux = aux[topN_val == 0]
                sc_idx[aux] = 0
                print('after nonmax remains', np.sum(sc_idx))
                # row_col = np.c_[np.unravel_index(idx, ss.shape)]
                centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1,1), yv.flatten()[out_idxs].reshape(-1,1)], axis=1)
            else:
                out_idxs = np.random.choice(idxs, n_curr_level, p=(mask_valid / mask_valid.sum()).flatten())
                centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1,1), yv.flatten()[out_idxs].reshape(-1,1)], axis=1)
        print('Used:', np.sum(centers[:,0] != -1), '/', centers.shape[0]) # these will be eliminated during LAF check (they lie outside)

        angles = np.random.uniform(0, np.pi, nLAFs)
        tilts = np.random.uniform(1.0, self.max_tilt, nLAFs)
        tilt_matrix = np.repeat(np.eye(2).reshape(1,2,2), nLAFs, axis=0)
        tilt_matrix[:,0,0] = np.sqrt(tilts)
        tilt_matrix[:,1,1] = np.sqrt(1.0 / tilts)
        A = rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, get_rotation_matrix(angles))) / np.sqrt(tilts).reshape(-1,1,1)
        # A are always identity ???
        LAFs = np.concatenate([scales.reshape(-1,1,1) * A, centers.reshape(-1,2,1)], axis=2)
        return LAFs, scales

    def generateLAFs_avgRes(self, nLAFs, imtower, border=5, mask=None, init_scale = 16, type='sum'):
        w, h = imtower[0].width, imtower[0].height
        scales = np.exp(np.random.uniform(np.log(init_scale), np.log(np.minimum(h, w) / 4.0 - 2 * border), (nLAFs)))
        centers = np.zeros((nLAFs, 2)) - 1
        # mask_pyr_list = [self.get_mask_pyr(imt, [mask, np.ones((h, w))][mask is None] ) for imt in imtower]
        mask_pyr_list = [get_mask_pyr(imt, [mask, np.ones((h, w))][mask is None], self.weight_function, self.scaleit) for imt in imtower]

        level = 0
        sc = init_scale
        while sc <= min(w, h) / 4.0:
            sc_idx = scales >= sc
            n_curr_level = sc_idx.sum()
            if n_curr_level == 0:
                sc = int(sc * 2)
                level += 1
                continue

            mask_list = []
            for masks in mask_pyr_list:
                h1, w1 = masks[level].shape
                assert (w1 > 2*border) and (h1 > 2*border), 'non fixed'
                mask_curr = masks[level][border : h1 - border, border : w1 - border]
                if mask_curr.sum() < 1e-10:
                    continue
                x = np.linspace(sc + border, w - 1 - (sc + border), w1 - 2 * border)
                y = np.linspace(sc + border, h - 1 - (sc + border), h1 - 2 * border)
                xv, yv = np.meshgrid(x, y)
                mask_list += [mask_curr]
            if type in ['sum']:
                mask_sum = sum(mask_list)
            elif type in ['median']:
                mask_sum = np.median(np.array(mask_list), 0)
            mask_sum /= mask_sum.sum()

            idxs = np.arange(0, xv.flatten()[:].shape[0])
            out_idxs = np.random.choice(idxs, n_curr_level, p=mask_sum.flatten())
            print('last_response: {}'.format(mask_sum.flatten()[out_idxs[-1]]))

            centers[sc_idx, :] = np.concatenate([xv.flatten()[out_idxs].reshape(-1, 1), yv.flatten()[out_idxs].reshape(-1, 1)], axis=1)
            sc = int(sc * 2)
            level += 1

        angles = np.random.uniform(0, np.pi, (nLAFs))
        tilts = np.random.uniform(1.0, self.max_tilt, (nLAFs))
        rot = get_rotation_matrix(angles)
        tilt_matrix = np.repeat(np.eye(2).reshape(1, 2, 2), nLAFs, axis=0)
        tilt_matrix[:, 0, 0] = np.sqrt(tilts)
        tilt_matrix[:, 1, 1] = np.sqrt(1.0 / tilts)
        A = rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1, 1, 1)
        LAFs = np.concatenate([scales.reshape(-1, 1, 1) * A, centers.reshape(-1, 2, 1)], axis=2)
        return LAFs, scales

    def generateLAFs_avgImg_new(self, imgs, mask=None, type='median', use_gauss=False):
        img_np = np.array([np.array(c) for c in imgs])
        if type in ['median']:
            img = np.median(img_np, 0)
        elif type in ['mean']:
            img = np.mean(img_np, 0)
        elif type in ['one']:
            img = img_np[0]
        img = img.astype(np.float) / 255
        print(img.shape)

        if self.sigmas_v in ['v1']:
            sigmas, base_size = [1.8**i for i in range(5)], 6
        elif self.sigmas_v in ['v12']:
            sigmas, base_size = [1.8**i for i in range(1, 6)], 12
        elif self.sigmas_v in ['v13']:
            sigmas, base_size = [1.8**i for i in range(0,5)], 24
        elif self.sigmas_v in ['v14']: # is 001, or (0,4,30)
            sigmas = [1.8**i for i in range(0,4)]
            base_size = 30
        elif self.sigmas_v in ['v15']:
            sigmas = [1.8**i for i in range(0,5)]
            base_size = 30
        elif self.sigmas_v in ['v16']:
            sigmas = [1.8**i for i in range(0,6)]
            base_size = 30
        elif self.sigmas_v in ['v17']:
            sigmas = [1.8**i for i in range(1,4)]
            base_size = 30
        elif self.sigmas_v in ['v2']:
            sigmas, base_size = [3,4,5,6,7], 6
        elif self.sigmas_v in ['v3']: # best so far
            sigmas = [5, 15, 30, 60, 200]
            base_size = 3
        elif self.sigmas_v in ['v32']:
            sigmas, base_size = [5, 15, 30, 60, 200], 6
        elif self.sigmas_v in ['v4']:
            sigmas, base_size = [5, 10, 15, 20, 25], 8
        elif self.sigmas_v in ['v5']:
            sigmas, base_size = [5, 15, 30, 60], 6
        elif self.sigmas_v in ['v6']:
            sigmas, base_size = [2,4,8,16,32], 6

        elif self.sigmas_v[0] in ['e']: # format exxx
            a = [0,1][int(self.sigmas_v[1])]
            b = [4,5][int(self.sigmas_v[2])]
            c = [24,30][int(self.sigmas_v[3])]
            print('abc:', a,b,c)
            sigmas, base_size = [1.8**i for i in range(a, b)], c
        else: assert False, 'Wrong sigmas_v'

        detector = HessDetector(sigmas=sigmas, base_size=base_size, fixed_MP=self.fixed_MP)
        nLAFs = sys.maxsize
        thr = self.init_thr / 2
        while nLAFs > self.max_sets_per_img:
            thr *= 2
            scales, centers = detector.detect_positions(img, mask, thr=thr)
            nLAFs = len(scales)
            print('detected', nLAFs, 'points', ', thr=', thr)
        # print('max x:',np.max(centers[:,0]))

        if use_gauss:
            ret_mask = detector.detect_positions(img, mask, thr=thr, ret_mask=True)
            scales, centers = gaussian_select(ret_mask, self.gauss_s, sigmas, base_size)
        # print('max x:',np.max(centers[:,0]))

        angles = np.random.uniform(0, np.pi, nLAFs)
        tilts = np.random.uniform(1, self.max_tilt, nLAFs)
        rot = get_rotation_matrix(angles)
        tilt_matrix = np.repeat(np.eye(2).reshape(1,2,2), nLAFs, axis=0)
        tilt_matrix[:,0,0] = np.sqrt(tilts)
        tilt_matrix[:,1,1] = np.sqrt(1.0 / tilts)
        A = rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1,1,1)
        LAFs = np.concatenate([scales.reshape(-1,1,1) * A, centers.reshape(-1,2,1)], axis=2)
        return LAFs, scales, detector, thr

    def generateLAFs_SIFT(self, imgs, mask=None, type='median'):
        imgs = np.array([np.array(c) for c in imgs])
        if type in ['median']:
            img = np.median(imgs, 0)
        elif type in ['mean']:
            img = np.mean(imgs, 0)
        elif type in ['one']:
            img = imgs[0]

        # img = Image.fromarray(img, 'L')
        # img.thumbnail(maxsize, Image.ANTIALIAS)
        # img = np.array(img)
        img = img.astype(np.uint8)

        n_features = int(1000.0*(np.prod(img.shape) / (640*480)))
        print('size', img.shape, 'features', n_features)

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
        kps, des = sift.detectAndCompute(img, mask=mask.astype(np.uint8))
        # img = cv.drawKeypoints(gray, kp, img)
        # cv2.imwrite('sift_keypoints.jpg', img)

        # angles = np.random.uniform(0, np.pi, nLAFs)
        # tilts = np.random.uniform(1, self.max_tilt, nLAFs)
        # rot = get_rotation_matrix(angles)
        # tilt_matrix = np.repeat(np.eye(2).reshape(1,2,2), nLAFs, axis=0)
        # tilt_matrix[:,0,0] = np.sqrt(tilts)
        # tilt_matrix[:,1,1] = np.sqrt(1.0 / tilts)
        # A = rectifyAffineTransformationUpIsUpNP(np.matmul(tilt_matrix, rot)) / np.sqrt(tilts).reshape(-1,1,1)
        angles = [c.angle*(2*np.pi/360) for c in kps]
        scales = np.array([12.0*c.size for c in kps])
        centers = np.array([np.array(c.pt) for c in kps])
        A = np.repeat(np.expand_dims(np.eye(2), 0), len(kps), 0)
        LAFs = np.concatenate([scales.reshape(-1,1,1) * A, centers.reshape(-1,2,1)], axis=2)
        return LAFs, angles

    def generateLAFs_HessBaum(self, imgs, type='median'):
        imgs = np.array([np.array(c) for c in imgs])
        if type in ['median']:
            img = np.median(imgs, 0)
        elif type in ['mean']:
            img = np.mean(imgs, 0)
        elif type in ['one']:
            img = imgs[0]
        img = torch.from_numpy(img.astype(np.float32)).cuda()

        n_features = int(1000.0*(np.prod(img.shape) / (640*480)))
        print('img_size', img.shape, 'features', n_features)

        # HA = ScaleSpaceAffinePatchExtractor(mrSize=5.192, num_features=n_features, border=5, num_Baum_iters=16, AffNet=AffineShapeEstimator(patch_size=32)).cuda()
        HA = ScaleSpaceAffinePatchExtractor(mrSize=5.192, num_features=n_features, border=5, num_Baum_iters=-1, AffNet=None).cuda()

        LAFs, resp = HA(img.unsqueeze(0).unsqueeze(0))
        LAFs = LAFs.data.cpu().numpy()

        # img = img.astype(np.uint8)
        # n_features = int(1000.0*(np.prod(img.shape) / (640*480)))
        # print('size', img.shape, 'features', n_features)
        #
        # sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
        # kps, des = sift.detectAndCompute(img, mask=mask.astype(np.uint8))
        # angles = [c.angle*(2*np.pi/360) for c in kps]
        # scales = np.array([10.0*c.size for c in kps])
        # centers = np.array([np.array(c.pt) for c in kps])
        # A = np.repeat(np.expand_dims(np.eye(2), 0), len(kps), 0)
        # LAFs = np.concatenate([scales.reshape(-1,1,1) * A, centers.reshape(-1,2,1)], axis=2)
        angles = None
        return LAFs, angles

    def generate_LAFs_and_patches_from_dir(self, dir_name, nLAFs):
        imgs = get_imgs(pjoin(self.root, dir_name), grayscale=True, depth_path='') # just get the centers
        mask = np.ones((imgs[0].height, imgs[0].width))
        if self.masks_dir is not None:
            mask = Image.open(pjoin(self.masks_dir, dir_name.split('_')[0]) + '.png')
            if (mask.width, mask.height) != (imgs[0].width, imgs[0].height):
                print('mask_size', (mask.width, mask.height), 'imgs_size', (imgs[0].width, imgs[0].height), '-> downsizing mask')
                mask.thumbnail((imgs[0].width, imgs[0].height), Image.ANTIALIAS)
            mask = np.array(mask) > 0
            printc.green('cam mask found')

        # scales_cats = None
        angles = None
        if self.patch_gen in ['oneRes']:
            LAFs, scales = self.generateLAFs_avgImg(nLAFs, imgs, border=self.border, mask=mask, mode='one')
        elif self.patch_gen in ['medianImg']:
            LAFs, scales = self.generateLAFs_avgImg(nLAFs, imgs, border=self.border, mask=mask, mode='median')
        elif self.patch_gen in ['meanImg']:
            LAFs, scales = self.generateLAFs_avgImg(nLAFs, imgs, border=self.border, mask=mask, mode='mean')
        elif self.patch_gen in ['sumRes']:
            LAFs, scales = self.generateLAFs_avgRes(nLAFs, imgs, border=self.border, mask=mask, type='sum')
        elif self.patch_gen in ['medianRes']:
            LAFs, scales = self.generateLAFs_avgRes(nLAFs, imgs, border=self.border, mask=mask, type='median')
        # elif self.patch_gen in ['combi']:
        #     LAFs, scales, patches_mask = self.generateLAFs_combi(nLAFs, imgs, border=self.border, mask=mask, type='mean')
        elif self.patch_gen in ['new']:
            LAFs, scales, detector, thr = self.generateLAFs_avgImg_new(imgs, mask=mask, type='mean', use_gauss=self.to_gauss)
        elif self.patch_gen.lower() in ['sift']:
            LAFs, angles = self.generateLAFs_SIFT(imgs, mask=mask, type='mean')
        elif self.patch_gen.lower() in ['hessbaum']:
            LAFs, angles = self.generateLAFs_HessBaum(imgs, type='mean') # mask not supported

        As = [get_A(laf, self.patch_size, imgs[0], spx=self.spx, spy=self.spy) for laf in LAFs]
        print('survived', sum([A is not None for A in As]), 'points')
        LAFs = LAFs[np.array([A is not None for A in As])]
        # scales = scales[np.array([A is not None for A in As])]
        As = [A for A in As if A is not None]

        if self.depths!='': # save with RGB for visualization (format RGBA)
            imgs = get_imgs(pjoin(self.root, dir_name), grayscale=False, depth_path=self.depths)
        else:
            if self.AMOS_RGB:
                imgs = get_imgs(pjoin(self.root, dir_name), grayscale=False, depth_path='')
            else:
                imgs = get_imgs(pjoin(self.root, dir_name), grayscale=True, depth_path='')

        patches = []
        if angles is None:
            angles = np.degrees(np.random.uniform(-np.pi, np.pi, len(LAFs)))
        p = pp.ProcessPool(multiprocessing.cpu_count())
        p.restart()
        for img in imgs:
            new_patches = p.map(crop_patch(img, self.patch_size), As, angles)
            patches += [torch.cat([(255 * tforms.ToTensor()(x).unsqueeze(0)).byte() for x in new_patches])]
        tracks = torch.stack(patches, dim=1)
        print('survived', tracks.shape[0], 'points')

        if self.patch_gen in ['new'] and self.use_patchmask:
            scales_and_centers = []
            for img in tqdm(imgs, desc='detecting on single images ...'):
                scales_and_centers += [detector.detect_positions(np.array(img).astype(np.float) / 255, mask, thr=thr)]
            scales_singles = [np.array(c[0]) for c in scales_and_centers]
            # safes_singles = np.array([safe_dist() for c in scales_singles])
            centers_singles = [np.array(c[1]) for c in scales_and_centers] # centers on single imgs
            # dists_and_centers = [(detector.safe_dist(s), b) for s,b in scales_and_centers]
            printc.green('updating patch mask ...')
            def fce(centers_single, scales_single):
                patches_mask = np.zeros((len(LAFs)), dtype=bool)
                dists = spatial.distance.cdist(LAFs[:,:,2], centers_single, 'euclidean')
                valid = np.sum(dists <= 10, 1) > 0 # at least one must be close
                patches_mask[valid] = 1
                return patches_mask
            patches_mask = p.map(fce, centers_singles, scales_singles) # parallel over imgs
            patches_mask = np.stack(patches_mask, axis=-1)
            # print('scales', patches_mask.shape)
            print(np.sum(patches_mask), '/', np.product(patches_mask.shape), 'patches active')
        else:
            patches_mask = np.ones((len(LAFs), len(imgs)), dtype=bool)
        p.close()

        # return tracks, LAFs, patches_mask, scales
        return tracks, LAFs, patches_mask

    def get_collisions(self):
        all_cams = np.unique(self.cam_idxs.numpy().astype(np.int)).tolist()
        cam_idxs = [np.array([i for i, x in enumerate(self.cam_idxs) if x == c]) for c in all_cams]

        printc.green('searching collisions')
        # for cur_sidxs in tqdm(cam_idxs, desc='searching collisions'):
        def wrap_fce(patch_size, LAFs):
            def fce(cur_sidxs):
                collisions = []
                points = np.array([get_points(patch_size, laf) for laf in LAFs[cur_sidxs]])
                for i, cur_sidx in enumerate(cur_sidxs):
                    collides = np.array([Polygon(points[i]).intersects(Polygon(x)) for x in points])
                    collisions += [cur_sidxs[collides]]
                return collisions
            return fce

        p = pp.ProcessPool(multiprocessing.cpu_count())
        p.restart()
        collisions = p.map(wrap_fce(self.patch_size, self.LAFs), cam_idxs)
        p.close()
        return list(itertools.chain.from_iterable(collisions))

    def get_collisions_(self, cam_idxs, LAFs):
        all_cams = np.unique(cam_idxs.numpy().astype(np.int)).tolist()
        cam_idxs = [np.array([i for i, x in enumerate(cam_idxs) if x == c]) for c in all_cams]

        printc.green('searching collisions')
        # for cur_sidxs in tqdm(cam_idxs, desc='searching collisions'):
        def wrap_fce(patch_size, LAFs):
            def fce(cur_sidxs):
                collisions = []
                points = np.array([get_points(patch_size, laf) for laf in LAFs[cur_sidxs]])
                for i, cur_sidx in enumerate(cur_sidxs):
                    collides = np.array([Polygon(points[i]).intersects(Polygon(x)) for x in points])
                    collisions += [cur_sidxs[collides]]
                return collisions
            return fce

        p = pp.ProcessPool(multiprocessing.cpu_count())
        p.restart()
        collisions = p.map(wrap_fce(self.patch_size, LAFs), cam_idxs)
        p.close()
        return list(itertools.chain.from_iterable(collisions))

    def prepare_dataset(self, Ntracks, overwrite=False):
        if os.path.exists(self.data_path) and not overwrite:
            printc.green('Loading', self.data_path, '...')
            self.data = torch.load(self.data_path)
            print('Found', len(self.data['patch_sets']), 'patch sets')
            self.patch_sets = self.data['patch_sets']
            if self.depths!='' and not self.AMOS_RGB:
                print('merging RGB channels')
                self.patch_sets = torch.stack([torch.mean(self.patch_sets[:,:,:3,:,:].float(),dim=2).byte(), self.patch_sets[:,:,3,:,:]], 2) # squash the RGB
            if self.only_D:
                print('only_d')
                self.patch_sets = self.patch_sets[:,:,-1,:,:].unsqueeze(2)
            self.LAFs       = self.data['LAFs'].data.cpu().numpy()
            self.cam_idxs   = self.data['cam_idxs']
            if 'collisions' in self.data.keys():
                self.collisions = self.data['collisions']
            self.patches_mask = np.ones((self.patch_sets.shape[0], self.patch_sets.shape[1]), dtype=bool)
            if 'patches_mask' in self.data.keys() and self.use_patchmask:
                print('patches_mask loaded')
                self.patches_mask = self.data['patches_mask']
                self.allowed_patches = np.arange(self.patch_sets.shape[0])[np.sum(self.patches_mask, 1) < self.Npositives]
            self.allowed_patches = np.ones((self.patch_sets.shape[0]), dtype=bool)
            if self.allowed_sets is not None:
                self.patch_sets = self.patch_sets[self.allowed_sets]
            return

        start_time = time.time()
        printc.red('Not found: {}'.format(self.data_path))
        printc.green('Making dataset ...')
        printc.red('Collisions detection:', ['NO, turn it on by --colls','YES (this takes considerate time at the end)'][self.use_collisions])

        img_dirs = sorted([x for x in os.listdir(self.root) if os.path.isdir(pjoin(self.root, x))])
        Npixels = sum([np.prod(Image.open(glob(pjoin(self.root, c, '*'))[0]).convert('RGB').size[:2]) for c in img_dirs])

        tracks, lafs, cam_idxs, idxs_good, patches_mask, scales, scale_cats = [], [], [], [], [], [], []
        for i, img_dir in enumerate(img_dirs):
            printc.yellow('----- dir {}/{}: {} -----'.format(i+1, len(img_dirs), img_dir))

            img_PIL = Image.open(glob(pjoin(pjoin(self.root, img_dir), '*'))[0]).convert('RGB')
            Nnew_tracks = int(Ntracks * (float(img_PIL.size[0] * img_PIL.size[1]) / Npixels))
            Nnew_tracks = max(Nnew_tracks, self.min_sets_per_img)
            print('img size equivalent to', Nnew_tracks, 'points')
            # new_tracks, new_lafs, new_patches_mask, new_scales = self.generate_LAFs_and_patches_from_dir(img_dir, Nnew_tracks)
            new_tracks, new_lafs, new_patches_mask = self.generate_LAFs_and_patches_from_dir(img_dir, Nnew_tracks)
            tracks      += [new_tracks]
            lafs        += [new_lafs]
            patches_mask+= [new_patches_mask]
            # scales      += [new_scales]
            cam_idxs    += [torch.ones(len(new_tracks)) * i]

        printc.yellow('----- finished -----\nmin # patches:', min([len(n) for n in tracks]))
        print('# patches total:', sum([c.shape[0] for c in tracks]))

        printc.green('concating data ...')
        out = {}
        out['patch_sets']   = torch.cat(tracks, dim=0) # this line fails in newer pytorch with segmentation fault (in my virtualenv)
        out['LAFs']         = torch.from_numpy(np.concatenate(lafs, axis=0)).float() #torch.cat(lafs, dim=0) torch.from_numpy(LAFs).float()
        # out['scales']       = torch.from_numpy(np.concatenate(scales)).float()
        out['cam_idxs']     = torch.cat(cam_idxs).long()
        if self.use_collisions:
            out['collisions']   = self.get_collisions_(out['cam_idxs'], out['LAFs'].data.cpu().numpy())
        out['view_names']   = list(map(lambda x: os.path.basename(x), img_dirs))
        if patches_mask[0] is not None: out['patches_mask'] = np.concatenate(patches_mask, axis=0)

        printc.green('saving datafile {} ...'.format(self.data_path))
        torch.save(out, open(self.data_path, 'wb'))
        print('Dataset created in {} seconds. Run again the same command to start training.'.format(int(time.time() - start_time)))
        exit(0)

def get_attrs_PS(PS):
    big_PS = int(1.5 * PS)
    return big_PS, (big_PS - PS) / 2, (big_PS + PS) / 2

def get_points(PS, LAF, aslist=True):
    big_PS, left, right = get_attrs_PS(PS)
    points = np.transpose(np.array([[left, left, 1], [left, right, 1], [right, right, 1], [right, left, 1]]))
    A = LAF2A(LAF, big_PS)
    if aslist:
        return [list(c) for c in list(np.transpose(A @ points))]
    return A @ points

def get_A(LAF, PS, img, spx=0, spy=0):
    big_PS, left, right = get_attrs_PS(PS)
    A = LAF2A(LAF, big_PS)

    if spx != 0 or spy != 0:  # random shift ... aplies to each patch individually based on its scale
        sx = int(2 * spx * A[0,0])
        sy = int(2 * spy * A[0,0])  # A[0,0] is the scale
        A[0,2] += random.randint(0, sx) - (sx / 2)
        A[1,2] += random.randint(0, sy) - (sy / 2)

    points = np.transpose(np.array([[left,left,1],[left,right,1],[right,right,1],[right,left,1]]))
    ps = A @ points
    wrong = np.sum(ps[0,:]<0) + np.sum(ps[0,:]>img.width-1) + np.sum(ps[1,:]<0) + np.sum(ps[1,:]>img.height-1)
    if wrong > 0: # falls outside image area
        return None
    return A

def crop_patch(img, PS):
    big_PS, left, right = get_attrs_PS(PS)
    def fce(A, ang):
        return img.transform((big_PS, big_PS), Image.AFFINE, A.ravel(), Image.BICUBIC).rotate(ang).crop((left, left, right, right))
    return fce

def get_imgs(dir_in, grayscale:bool, depth_path=''):
    files = os.listdir(dir_in)
    img_fnames = sorted([x for x in files if os.path.splitext(x)[1].strip('.') in ['jpg', 'jpeg', 'png', 'ppm', 'bmp']])
    trans_fnames = sorted([x for x in files if os.path.splitext(x)[1] == ''])
    images = []
    assert (len(trans_fnames) + 1 == len(img_fnames)) or (len(trans_fnames) == 0)
    for i in range(len(img_fnames)):
        img = Image.open(pjoin(dir_in, img_fnames[i]))

        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        if depth_path != '': # not working now, how to represent RGB-D image to thumbnail?
            aux = pjoin(dir_in, img_fnames[i])
            depth_name = pjoin(depth_path,
                                      os.path.basename(os.path.dirname(aux)),
                                      os.path.basename(aux))
            img_d = Image.open(depth_name).convert('L').resize(img.size, Image.ANTIALIAS)
            img_d = np.expand_dims(np.array(img_d), 2)
            img = np.concatenate([np.array(img), img_d], 2)
            # print(np.array(img).shape)
            img = Image.fromarray(img, 'RGBA')
            # print(np.array(img_d).shape)
            # input()
        if i != 0:
            if len(trans_fnames) > 0:
                h_fname = pjoin(dir_in, trans_fnames[i - 1])
                H = np.loadtxt(h_fname)
                H /= H[2, 2]
                if np.abs(H - np.eye(3)).max() > 1e-5:  # do transform
                    img = img.transform((img.width, img.height), Image.PERSPECTIVE, H.ravel(), Image.BICUBIC)
            else:
                assert (img.width == images[0].width) and (img.height == images[0].height)
        images += [img]
    return images

def get_mask_pyr(img, mask, WF, scaleit=False):
    mask_pyr, pyr = [], [deepcopy(img)]
    w1, h1 = img.width, img.height
    curr_mask = Image.fromarray(np.uint8(mask * 255))
    scale = 1.0
    while min(w1, h1) > 10:
        w1, h1 = pyr[-1].size
        act_img = pyr[-1].copy()
        mask_pyr.append(WF(act_img, scale) * np.array(curr_mask))
        act_img.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
        curr_mask.thumbnail((w1 / 2, h1 / 2), Image.ANTIALIAS)
        pyr.append(act_img)
        if scaleit:
            scale *= 2
    return mask_pyr

def get_random_subset(source, num):
    # num <= 0 turns off selection
    if num <= 0:
        return source
    return list(random.sample(source, num))
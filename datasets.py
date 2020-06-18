from copy import *
import torch.utils.data as data
from Utils.AMOS_dataset import AMOS_dataset
from Learning.handcrafted_modules import *
from utils_ import *
from Learning.learning import *
from dataclasses import dataclass
import kornia.augmentation.functional as KF
import numpy_indexed as npi


def get_test_loaders(names, bsize):
    aux = [
        torch.utils.data.DataLoader(
            TestLoader_6Brown(root=os.path.join('Datasets/6Brown'), name=name, transform=trans_resize32),
            batch_size=bsize, shuffle=False, **{'num_workers': 4, 'pin_memory': True})
        for name in names
    ]
    for a in aux: # for fastai2
        a.to = lambda x: None
    return aux

class TestLoader_6Brown:
    def __init__(self, transform=None, name=None, root=None):
        self.name = name
        self.data_file = os.path.join(root, '{}.pt'.format(name))
        self.transform = transform

        print('# Found cached data {}'.format(self.data_file))
        dataset = torch.load(self.data_file)
        if type(dataset) != type({}):
            assert False, 'the format has changed, I expect dictionary'
        self.patches = dataset['patches']
        self.matches = dataset['matches']
        self.labels = dataset['labels']
        if 'cam_ids' in dataset.keys():
            self.cam_ids = dataset['cam_ids']

    def __getitem__(self, index):
        m = self.matches[index]
        img1 = safe_transform(self.patches[m[0]], self.transform)
        img2 = safe_transform(self.patches[m[1]], self.transform)
        return img1, img2, m[2]

    def __len__(self):
        return self.matches.size(0)


@dataclass
class TrainLoader_6Brown(data.Dataset):
    dataset_path: str
    label_offset: int = 0
    fliprot: bool = False
    transform: object = None
    Npositives:int = 2
    args:object = None

    def __post_init__(self):
        super(TrainLoader_6Brown, self).__init__()

        print('loading',self.dataset_path)
        dataset = torch.load(self.dataset_path)
        if type(dataset) != type({}):
            assert False, 'the format has changed, I expect dictionary'
        self.patches = dataset['patches']
        self.labels = dataset['labels']
        if 'cam_ids' in dataset.keys():
            self.cam_ids = dataset['cam_ids']

        if len(self.patches.shape) == 4 and self.patches.shape[-1]==3:
            self.patches = self.patches.permute(0,3,1,2)

        print('found patches:',len(self.patches))
        print('found classes:',len(torch.unique(self.labels)))
        self.labels += self.label_offset

    def max_label(self):
        return torch.max(self.labels).item()
    def min_label(self):
        return torch.min(self.labels).item()
    def num_classes(self):
        return len(set(self.labels.data.cpu()))
    def get_labels(self):
        return self.labels.data.cpu().numpy()
    def __len__(self):
        return len(self.tuples)
    def to(self, dummy): # for fastai
        return

    def generate_tuples(self, bs_seq):
        indices = dict()
        for idx, label in enumerate(self.labels):
            dict_add(indices, key=label.item(), value=idx, acc='list')

        if self.args.fewcams:
            print('fewcams')
            gb = npi.group_by(self.cam_ids)
            allowed_cams = torch.unique(self.cam_ids)
            gr_labels = gb.split_array_as_list(self.labels)

            per_cam = {}
            for a, b in zip(allowed_cams, gr_labels):
                per_cam[a] = np.unique(b)

            self.tuples = []
            for batch_size in tqdm(bs_seq, desc='Generating tuples'):
                cur_cams = list(per_cam.keys())
                cs = set()
                while True:
                    cam = random.choice(cur_cams)
                    cur_cams.remove(cam)
                    for c in per_cam[cam]:
                        cs.add(c)
                        if len(cs) == batch_size:
                            break
                    if len(cs) == batch_size:
                        break
                for c1 in cs:
                    if len(indices[c1]) == 1:
                        ns = [indices[c1][0]] * 2  # the same patch twice, this is fix for classes of size 1
                    elif len(indices[c1]) == 2 or len(indices[c1]) <= self.Npositives:
                        ns = indices[c1]
                    else:
                        ns = random.sample(indices[c1], self.Npositives)
                    self.tuples += [(c1, ns)]
            return

        if self.args.fewcams_dups: # this is wrong implementation, but was slightly better?
            print('fewcams_dups')
            gb = npi.group_by(self.cam_ids)
            allowed_cams = torch.unique(self.cam_ids)
            gr_labels = gb.split_array_as_list(self.labels)

            per_cam = {}
            for a, b in zip(allowed_cams, gr_labels):
                per_cam[a] = np.unique(b)

            self.tuples = []
            for batch_size in tqdm(bs_seq, desc='Generating tuples'):
                cur_batch_size = batch_size
                cur_cams = list(per_cam.keys())
                while True:
                    cam = random.choice(cur_cams)
                    cur_cams.remove(cam)
                    cs = per_cam[cam]
                    cs = random.sample(list(cs), min(len(cs), cur_batch_size))
                    cur_batch_size -= len(cs)
                    if cur_batch_size <= 0:
                        break
                    for c1 in cs:
                        if len(indices[c1]) == 1:
                            ns = [indices[c1][0]] * 2  # the same patch twice, this is fix for classes of size 1
                        elif len(indices[c1]) == 2 or len(indices[c1]) <= self.Npositives:
                            ns = indices[c1]
                        else:
                            ns = random.sample(indices[c1], self.Npositives)
                        self.tuples += [(c1, ns)]
            return

        self.tuples = []
        set_labels = set(self.labels.data.cpu().numpy())
        for batch_size in tqdm(bs_seq, desc='Generating tuples'):
            cs = random.sample(set_labels, batch_size) # no replace
            for c1 in cs:
                if len(indices[c1]) == 1:
                    ns = [indices[c1][0]]*2 # the same patch twice, this is fix for classes of size 1
                elif len(indices[c1]) == 2 or len(indices[c1]) <= self.Npositives:
                    ns = indices[c1]
                else:
                    ns = random.sample(indices[c1], self.Npositives)
                self.tuples += [(c1, ns)]

    def __getitem__(self, idx):
        t = self.tuples[idx] # t[0] label, t[1] indices
        patches = [safe_transform(self.patches[i], self.transform) for i in t[1]]
        # if len(patches[0].shape) == 2:
        #     patches = [c.unsqueeze(0) for c in patches]

        if self.fliprot:
            if random.random() > 0.5:  # do rotation
                patches = [p.permute(0, 2, 1) for p in patches]
            if random.random() > 0.5:  # do flip
                patches = [torch.from_numpy(deepcopy(c.numpy()[:, :, ::-1])) for c in patches]
        return {'data': torch.stack(patches, 0), 'labels': torch.tensor(t[0]).repeat(len(patches))}


class DS_parent:
    def __init__(self):
        self.group_ids = [0]

@dataclass
class DS_Brown(DS_parent):
    path:str
    transform:object
    fliprot:bool = True
    workers:int = 1

@dataclass
class DS_AMOS(DS_parent):
    dir:str
    data_name:str
    patch_sets:int
    weight_function:object
    transform:object
    patch_gen:str
    cams_in_batch:int
    fliprot:bool = True
    masks_dir:str = None
    workers:int = 2
    data_path:str = None

@dataclass
class DS_wrapper:
    datasets:[DS_parent]
    tuples:int
    batch_size:int
    frequencies:[int] = None
    fliprot:bool = False
    collate_fn:object = my_collate_fn
    Npositives:int = 2

    def __post_init__(self):
        assert len(self.datasets)>0, 'no datasets, check your setup'
        self.gid_to_DS = {}
        self.fixed_batch = False # used for lr_find
        if self.frequencies is None:
            self.frequencies = [1]*len(self.datasets)

        self.bad_patches = []
        self.good_patches = []

        self.ds_idx_seq = []
        self.loaders = []

    def prepare_epoch(self, gen_tuples=True, good_patches = None):
        if len(self.loaders) == 0: # create only once
            self.loaders_info = {}
            label_offset = 0
            for i,d in enumerate(self.datasets):
                if isinstance(d, DS_Brown):
                    self.loaders += [
                        TrainLoader_6Brown(
                            dataset_path=os.path.abspath(d.path),
                            fliprot=d.fliprot,
                            transform=d.transform,
                            label_offset=label_offset,
                            Npositives=self.Npositives,
                            args=self.args,
                        )
                    ]
                elif isinstance(d, DS_AMOS):
                    # new_split_name = d.split_name+ '_masks:' + ('None' if d.masks_dir is None else os.path.basename(d.masks_dir))
                    self.loaders += [
                        AMOS_dataset(
                            root_dir=d.dir,
                            # data_name=new_split_name,
                            data_name=d.data_name,
                            data_path=d.data_path,
                            Npatch_sets=d.patch_sets,
                            masks_dir = self.args.masks_dir,
                            weight_function=get_WF_from_string(d.weight_function),
                            Ntuples=self.tuples,
                            fliprot=d.fliprot,
                            transform=d.transform,
                            cams_in_batch=d.cams_in_batch,
                            patch_gen=d.patch_gen,
                            label_offset=label_offset,
                            separ_batches=self.args.separ_batches,
                            good_pr=self.args.good_pr,
                            pairs_imgs = self.args.pairs_imgs,
                            min_sets_per_img = self.args.min_sets_per_img,
                            max_sets_per_img = self.args.max_sets_per_img,
                            good_patches = good_patches,
                            new_batches = self.args.new_batches,
                            use_patchmask = self.args.use_patchmask,
                            use_collisions = self.args.use_collisions,
                            nonmax = self.args.nonmax,
                            sigmas_v = self.args.sigmas_v,
                            init_thr = self.args.init_thr,
                            scaleit = self.args.scaleit,
                            thrit = self.args.thrit,
                            to_gauss = self.args.to_gauss,
                            gauss_s = self.args.gauss_s,
                            fixed_MP = self.args.fixed_MP,
                            depths = self.args.depths,
                            AMOS_RGB = self.args.AMOS_RGB,
                            only_D = self.args.only_D,
                            Npositives=self.Npositives,
                        )
                    ]
                    self.amos = self.loaders[-1]
                    self.amos.bad_patches = self.bad_patches
                else:
                    raise Exception('invalid DS class')
                self.loaders[-1].bs = self.batch_size
                self.loaders[-1].ds_info = {'workers':d.workers, 'dataset':d}
                label_offset = self.loaders[-1].max_label()+1

        self.total_num_labels = sum([c.num_classes() for c in self.loaders])

        if not gen_tuples:
            return

        self.ds_idx_seq = np.random.choice(list(range(len(self.datasets))), size=self.n_iters(), p=self.frequencies) # prepare sequence of groups for the epoch
        if self.args.combine in ['epoch']:
            print('combine: EPOCH')
            self.ds_idx_seq = []
            for iii in range(len(self.datasets)):
                self.ds_idx_seq += [iii] * int(self.n_iters() / len(self.datasets))
                # self.ds_idx_seq = self.n_iters() / 3 # prepare sequence of groups for the epoch
        if self.args.combine in ['inbatch']:
            print('combine: inbatch')
            self.ds_idx_seq = []
            for ii in range(int(self.n_iters() / len(self.datasets))):
                self.ds_idx_seq += list(range(len(self.datasets)))

        self.iters = []
        for i,l in enumerate(self.loaders):
            l.generate_tuples(bs_seq=[l.bs for c in self.ds_idx_seq if c == i])
            self.iters += \
                [iter(torch.utils.data.DataLoader(l, batch_size=l.bs, shuffle=False, num_workers=self.loaders[i].ds_info['workers'],
                                                  pin_memory=True, collate_fn=my_collate_fn))]

    def init(self, model_dir, save_name, args=None):
        self.args = args
        if not len(self.frequencies)==len(self.datasets):
            raise Exception('must be len(frequencies)==len(datasets), that is one relative frequency for each DS')
        self.frequencies = np.array(self.frequencies) / np.sum(np.array(self.frequencies))

        for ds in self.datasets:
            printc.yellow(str(sorted(ds.__dict__.items())))

        self.prepare_epoch(gen_tuples=False)

        if save_name is not None: # LossNet sets to None
            os.makedirs(pjoin(model_dir, save_name), exist_ok=True)
            with open(pjoin(model_dir, save_name, 'setup.txt'), 'w') as f:
                print('args:\n{}\n'.format(str(args)), file=f)
                for l,d in zip(self.loaders, self.datasets):
                    if isinstance(d, DS_AMOS):
                        d.data_path = l.data_path
                    # print(sorted(d.__dict__.items()), file=f)
                    print(str(type(d))+ ':\n' + '\n'.join([str(x) for x in sorted(d.__dict__.items())]) + '\n', file=f)

        self.sigmas = {} # sigmas for each AMOS dataset - cannot be kept in loader, which is recreated each epoch
        for i,l in enumerate(self.loaders):
            if isinstance(l, AMOS_dataset):
                self.sigmas[i] = torch.ones((len(l.patch_sets)), requires_grad=True, device='cuda')
                torch.nn.init.normal_(self.sigmas[i], mean=1, std=1)
        return self

    def n_iters(self):
        return int(self.tuples / self.batch_size)
    def __len__(self):
        return len(self.ds_idx_seq)
    def max_label(self):
        return max([l.max_label() for l in self.loaders])

    def __getitem__(self, idx):
        if idx==0 and self.fixed_batch:
            printc.red('FIXED BATCH:', self.fixed_batch)
        if self.fixed_batch:
            idx = 0
        if idx >= len(self.ds_idx_seq):
            for iter in set(self.iters): # dataloaders
                iter.__del__()
            raise StopIteration
        ds_idx = self.ds_idx_seq[idx]

        if self.args.combine in ['inbatch']: ### destroy this later?
            if idx*len(self.datasets) >= len(self.ds_idx_seq):
                for iter in set(self.iters):  # dataloaders
                    iter.__del__()
                raise StopIteration

            aux = []
            for l in self.iters:
                sample = next(l)
                aux += [sample]

            info = {}
            info['labels'] = torch.cat([c['labels'] for c in aux])
            data = torch.cat([c['data'] for c in aux]).float().cuda()
            return data, info

        info = {}
        loader = self.loaders[ds_idx]
        sample = next(self.iters[ds_idx])
        info['loader'] = 'other'
        if isinstance(self.loaders[ds_idx].ds_info['dataset'], DS_AMOS): # only AMOS provides info
            if hasattr(loader, 'block_sizes'):
                info['block_sizes'] = loader.block_sizes[0]
                loader.block_sizes = loader.block_sizes[1:]
            info['PS_idxs'] = sample['PS_idxs']
            info['set_idxs'] = sample['set_idxs']
            info['loader'] = 'AMOS'
            info['sigmas'] = self.sigmas[ds_idx][info['PS_idxs']]

        info['labels'] = sample['labels']
        info['ds_idx'] = ds_idx
        data = sample['data'].float().cuda()

        if self.args.duplicates>0:
            dups = self.args.duplicates
            info['labels'] = None
            dup_data = data[-2*dups:]
            data = torch.cat([data, dup_data])
        if self.args.antiaug:
            info['labels'] = None

            if not hasattr(self, 'antiaug_tr_pad'):
                self.antiaug_tr_pad = nn.Sequential(torch.nn.ReplicationPad2d(20)).cuda()
            if not hasattr(self, 'antiaug_tr_crop'):
                self.antiaug_tr_crop = nn.Sequential(kornia.augmentation.CenterCrop(64)).cuda()

            data_ = data.clone()
            data_ = self.antiaug_tr_pad(data_)
            As = data_[0::2]
            Ps = data_[1::2]

            B, CH, H, W = As.shape
            aff_dict = KF.rg.random_affine_generator(B, H, W,
                                                     degrees=(0.0, 0.0),
                                                     translate=(0.03, 0.03),
                                                     scale=(1.0, 1.0),
                                                     shear=(0.0, 0.0),
                                                     same_on_batch=False)
            aff_params = {}
            for k, v in aff_dict.items():
                aff_params[k] = v
                if k == 'translate':
                    aff_params[k] = v.to(v.device) + 0.03*torch.sign(v).to(v.device)
            antips = torch.zeros(*data_.shape).cuda()
            antips[0::2] = KF.apply_affine(As, aff_params)
            antips[1::2] = KF.apply_affine(Ps, aff_params)

            antips = self.antiaug_tr_crop(antips)
            data = torch.cat([data, antips])
        if self.args.K:
            data = {'data':data, 'loader':info['loader']}
        return data, info

    # def keep_info(self, losses, sets):
    #     losses = np.stack(losses, axis=0).flatten()
    #     sets = np.stack(sets, axis=0).flatten()
    #
    #     un_sets, mean_losses = npi.group_by(sets).mean(losses)
    #     print('min,max loss:', np.min(mean_losses), np.max(mean_losses))
    #     self.bad_patches = un_sets[(mean_losses>1.15).nonzero()[0]]

    def to(self, dummy):
        return

anti_aug_trans = transforms.Compose([ # CPU
    transforms.ToPILImage(),
    transforms.Pad(20, padding_mode='reflect'), # otherwise small black corners appear
    transforms.RandomResizedCrop(64, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor()
])

def get_train_dataset(args, data_name): # returns dataset wrapper
    AllBrown = [DS_Brown('Datasets/6Brown/' + c +'.pt', trans_resize32) for c in ('liberty', 'liberty_harris', 'notredame', 'notredame_harris', 'yosemite', 'yosemite_harris')]
    wrapper = None
    if args.ds in ['mix']:
        DSs = []
        DSs += AllBrown
        DSs += [DS_Brown('Datasets/HPatches/hpatches_split_view_train.pt', trans_resize32)]
        DSs += [DS_AMOS('Datasets/AMOS-views/AMOS-views-v3', data_name, args.patch_sets, args.weight_function, trans_AMOS,
                        args.patch_gen, args.cams_in_batch, workers=10)]
        wrapper = DS_wrapper(DSs, args.tuples, args.batch_size, frequencies=[1,1,1,1,1,1,6,6])
    elif args.ds in ['mix_good']:
        DSs = [
            DS_Brown('Datasets/HPatches/hpatches_illum-view_easy.pt', trans_resize32),
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
            DS_Brown('Datasets/Phototourism/hagia_sophia_interior_middleedges.pt', trans_resize32, workers=3),
        ]
    elif args.ds in ['lib+sofia+AMOS']:
        DSs = [
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
            DS_Brown('Datasets/Phototourism/hagia_sophia_interior_middleedges.pt', trans_resize32, workers=3),
            DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, trans_AMOS, args.patch_gen, args.cams_in_batch, workers=10),
        ]
    elif args.ds in ['lib+colo']:
        DSs = [
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
            DS_Brown('Datasets/Phototourism/colosseum_exterior_p1600000.pt'.format(args.ds), trans_resize32, workers=3),
        ]
    elif args.ds in ['v4+lib']:
        DSs = [
            DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, trans_AMOS, args.patch_gen, args.cams_in_batch, workers=10),
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
        ]
    elif args.ds in ['lib+v4']:
        DSs = [
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
            DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, trans_AMOS, args.patch_gen, args.cams_in_batch, workers=10),
        ]
    elif args.ds in ['lib+notre']:
        DSs = [
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
            DS_Brown('Datasets/6Brown/notredame.pt', trans_resize32),
        ]
    elif args.ds in ['lib+colo+notre']:
        DSs = [
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
            DS_Brown('Datasets/Phototourism/colosseum_exterior_p1600000.pt'.format(args.ds), trans_resize32, workers=3),
            DS_Brown('Datasets/6Brown/notredame.pt', trans_resize32),
        ]
    elif args.ds in ['lib+colo+notre_RGB']:
        DSs = [
            DS_Brown('Datasets/6Brown/liberty3x.pt', trans_resize32),
            DS_Brown('Datasets/Phototourism/colosseum_exterior_p1600000_RGB.pt'.format(args.ds), trans_resize32, workers=3),
            DS_Brown('Datasets/6Brown/notredame3x.pt', trans_resize32),
        ]
    elif args.ds in ['v4+lib+colo']:
        DSs = [
            DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, trans_AMOS, args.patch_gen, args.cams_in_batch, workers=10),
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
            DS_Brown('Datasets/Phototourism/colosseum_exterior_p1600000.pt'.format(args.ds), trans_resize32, workers=3),
        ]
    elif args.ds in ['v4+lib+colo+trevi']:
        DSs = [
            DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, trans_AMOS, args.patch_gen, args.cams_in_batch, workers=10),
            DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32),
            DS_Brown('Datasets/Phototourism/colosseum_exterior_p1600000.pt'.format(args.ds), trans_resize32, workers=3),
            DS_Brown('Datasets/Phototourism/trevi_fountain_p1600000.pt'.format(args.ds), trans_resize32, workers=3),
        ]
    elif args.ds in ['AB']:
        DSs = []
        DSs += AllBrown
        DSs += [DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, trans_AMOS, args.patch_gen, args.cams_in_batch, workers=10)]
        wrapper = DS_wrapper(DSs, args.tuples, args.batch_size, frequencies=[1, 1, 1, 1, 1, 1, 6])
    elif args.ds in ['brown']:
        wrapper = DS_wrapper(AllBrown, args.tuples, args.batch_size)

    elif args.ds in ['HP_easy']:
        DSs = [DS_Brown('Datasets/HPatches/hpatches_illum-view_easy.pt', trans_resize32, workers=3)]
    elif args.ds in ['HP_hard']:
        DSs = [DS_Brown('Datasets/HPatches/hpatches_illum-view_hard.pt', trans_resize32, workers=3)]
    elif args.ds in ['HP_tough']:
        DSs = [DS_Brown('Datasets/HPatches/hpatches_illum-view_tough.pt', trans_resize32, workers=3)]
    elif args.ds in ['HPs_illum_all']:
        DSs = [DS_Brown(f'Datasets/HPatches/{args.ds}.pt', trans_resize32, workers=3)]
    elif args.ds in ['HPs_illum_all-crop']:
        DSs = [DS_Brown('Datasets/HPatches/HPs_illum_all.pt', trans_crop, workers=3)]

    elif args.ds in ['liberty48']:
        DSs = [DS_Brown('Datasets/6Brown/liberty.pt', transform=trans_resize48, workers=8)]
    elif args.ds in ['liberty64']:
        DSs = [DS_Brown('Datasets/6Brown/liberty.pt', transform=None, workers=8)]
    elif args.ds in ['liberty','notredame','yosemite',
                     'yosemite_small','yosemite_cl',
                     'yosemite_cl_times2','yosemite_cl_times3','yosemite_cl_times4','yosemite_cl_times8']:
        DSs = [DS_Brown(f'Datasets/6Brown/{args.ds}.pt', transform=trans_resize32, workers=8)]
    elif args.ds in ['PS']:
        DSs = [DS_Brown(c, trans_crop, workers=1) for c in glob(pjoin('Datasets/PS-Dataset_trans', '*'))]

    elif args.ds in ['buckingham_palace','st_peters_square','trevi_fountain','hagia_sophia_interior','brandenburg_gate','colosseum_exterior','prague_old_town_square']:
        DSs = [DS_Brown(f'Datasets/Phototourism/{args.ds}.pt', trans_resize32, workers=3)]

    elif args.ds in ['trevi_fountain_p25000','trevi_fountain_p50000','trevi_fountain_p100000','trevi_fountain_p200000','trevi_fountain_p400000',
                     'trevi_fountain_p800000','trevi_fountain_p1600000','trevi_fountain_p3200000','trevi_fountain_p6400000',
                     'colosseum_exterior_p1600000_lower','colosseum_exterior_p1600000_upper',
                     'trevi_fountain_p1600000','trevi_fountain_p1600000_lower','trevi_fountain_p1600000_upper']:
        DSs = [DS_Brown(f'Datasets/Phototourism/{args.ds}.pt', trans_resize32, workers=3)]

    elif args.ds in ['colosseum_exterior_p1600000_RGB']:
        DSs = [DS_Brown(f'Datasets/Phototourism/{args.ds}.pt', trans_resize32, workers=3)]

    elif args.ds in ['hagia_sophia_interior_p1600000','brandenburg_gate_p1600000','colosseum_exterior_p1600000','buckingham_palace_p1600000','prague_old_town_square_p1600000']:
        DSs = [DS_Brown(f'Datasets/Phototourism/{args.ds}.pt', trans_resize32, workers=3)]

    elif args.ds in ['PS+lib']:
        DSs = [DS_Brown(c, trans_resize32, workers=1) for c in glob(pjoin('Datasets/PS-Dataset_trans', '*'))]
        DSs += [DS_Brown('Datasets/6Brown/liberty.pt', trans_resize32)]
        DSs += [DS_Brown('Datasets/6Brown/liberty_harris.pt', trans_resize32)]
        wrapper = DS_wrapper(DSs, args.tuples, args.batch_size, [1]*(len(DSs)-2)+[(len(DSs)-2)/2]*2)
    elif args.ds in ['v3','v4','v5']:
        DSs = [DS_AMOS(f'Datasets/AMOS-views/AMOS-views-{args.ds}', data_name, args.patch_sets, args.weight_function, trans_AMOS, args.patch_gen, args.cams_in_batch, workers=10)]
    elif args.ds in ['TILDE']:
        DSs = [DS_AMOS('Datasets/Related_work_AMOS', data_name, args.patch_sets, args.weight_function, trans_AMOS, args.patch_gen, args.cams_in_batch, workers=10)]
    elif args.ds in ['v4noshift']:
        DSs = [DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, trans_crop_resize, args.patch_gen, args.cams_in_batch, workers=10)]
    elif args.ds in ['v4crop']:
        DSs = [DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, transform_AMOS_s, args.patch_gen, args.cams_in_batch, workers=10)]
    elif args.ds in ['v4crop']:
        DSs = [DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, trans_crop, args.patch_gen, args.cams_in_batch, workers=10)]
    # elif args.ds in ['v4alb']:
    #     DSs = [DS_AMOS('Datasets/AMOS-views/AMOS-views-v4', data_name, args.patch_sets, args.weight_function, transform_AMOS_alb,
    #                    args.patch_gen, args.cams_in_batch, workers=10)]
    #     wrapper = DS_wrapper(DSs, args.tuples, args.batch_size, [1])
        wrapper = DS_wrapper(DSs, args.tuples, args.batch_size)
    else:
        assert False, 'unknown dataset, specify in datasets.py first'

    if wrapper is None:
        wrapper = DS_wrapper(DSs, args.tuples, args.batch_size)
    wrapper.Npositives = args.Npos
    if args.K != '': # kornia transform will be invoked inside of model
        printc.red('removing transforms in dataloaders, Kornia in models is turned on')
        for d in wrapper.datasets:
            d.transform = None
    return wrapper
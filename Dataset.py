import numpy as np
import torch, random
import torchvision.datasets as dset
from io_helpers import *
from copy import *
import torch.utils.data as data
from enum import Enum
from tqdm import tqdm
from WBSDataset import WBSDataset



class TripletPhotoTour(dset.PhotoTour):
    # From the PhotoTour Dataset it generates triplet samples
    # note: a triplet is composed by a pair of matching images and one of different class.

    mean = {
        "notredame": 0.4854,
        "yosemite": 0.4844,
        "liberty": 0.4437,
        "hpatches_dataset": 0.4210,
        "amos": 0.0,
        "amos_10K": 0.0,
        "AMOS_another_split_10K": 0.0,
        "notredame_harris": 0.4854,
        "yosemite_harris": 0.4844,
        "liberty_harris": 0.4437,
    }
    std = {
        "notredame": 0.1864,
        "yosemite": 0.1818,
        "liberty": 0.2019,
        "hpatches_dataset": 0.2245,
        "amos": 0.0,
        "amos_10K": 0.0,
        "AMOS_another_split_10K": 0.0,
        "notredame_harris": 0.1864,
        "yosemite_harris": 0.1818,
        "liberty_harris": 0.2019,
    }
    lens = {
        "notredame": 468159,
        "yosemite": 633587,
        "liberty": 450092,
        "hpatches_dataset": 1659944,
        "amos": 0.0,
        "amos_10K": 0.0,
        "AMOS_another_split_10K": 0.0,
        "liberty_harris": 379587,
        "yosemite_harris": 450912,
        "notredame_harris": 325295,
    }

    def __init__(self, train=True, transform=None, n_triplets=1000, batch_size=None, load_random_triplets=False, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        if self.train:
            print("Generating {} triplets".format(self.n_triplets))
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


class TotalDatasetsLoader(data.Dataset):
    def __init__(self, dataset_path, train=True, batch_size=None, fliprot=False, transform_dict=None, group_ids=[0], label_offset=0, *arg, **kw):
        super(TotalDatasetsLoader, self).__init__()
        dataset = torch.load(dataset_path)
        data, labels = dataset[0], dataset[1]
        labels += label_offset
        try:
            self.all_txts = dataset[-1]  # 'e1, e2, ...'
            if type(self.all_txts) != type(["some_text"]):
                raise Exception()
        except:
            self.all_txts = [""] * len(labels)

        # self.split_idxs = [-1] + [torch.max(labels).item()]
        # intervals = [(self.split_idxs[i], self.split_idxs[i + 1]) for i in range(len(self.split_idxs) - 1)]
        # self.range_intervals = [list(range(c[0] + 1, c[1] + 1)) for c in intervals]
        self.interval = list(range(label_offset, torch.max(labels).item()))

        self.data, self.labels = data, labels
        self.transform_dict = transform_dict
        self.train = train
        self.batch_size = batch_size
        self.fliprot = fliprot
        self.group_ids = group_ids
        self.label_offset = label_offset

    def max_label(self):
        return torch.max(self.labels).item()

    def generate_tuples(self):
        def create_indices():
            inds = dict()
            for idx, ind in enumerate(self.labels):
                ind = ind.item()
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        indices = create_indices()

        self.tuples = []
        for batch_size in tqdm(self.bs_seq, desc='generating tuples'):
            # print(batch_size)
            # cs = random.sample(self.range_intervals[0], batch_size) # no replace
            cs = random.sample(self.interval, batch_size) # no replace
            for c1 in cs:
                if len(indices[c1]) == 2:
                    n1, n2 = 0, 1
                else:
                    n1, n2 = random.sample(range(len(indices[c1])), 2) # this is like 10 times faster, what the heck numpy?
                    # n1, n2 = np.random.choice(range(len(indices[c1])), size=2, replace=False)
                self.tuples += [[indices[c1][n1], indices[c1][n2], -1]]

    def __getitem__(self, idx):
        def transform_img(img, transformation=None):
            return transformation(img) if transformation is not None else img

        t = self.tuples[idx]
        a, p = self.data[t[0]], self.data[t[1]]  # t[2] would be negative, not used

        img_a = transform_img(a, self.transform_dict[self.all_txts[t[0]]] if self.all_txts[t[0]] in self.transform_dict.keys() else self.transform_dict["default"])
        img_p = transform_img(p, self.transform_dict[self.all_txts[t[1]]] if self.all_txts[t[1]] in self.transform_dict.keys() else self.transform_dict["default"])

        if self.fliprot:  # transform images if required
            if random.random() > 0.5:  # do rotation
                img_a = img_a.permute(0, 2, 1)
                img_p = img_p.permute(0, 2, 1)
            if random.random() > 0.5:  # do flip
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:, :, ::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:, :, ::-1]))

        return img_a, img_p

    def __len__(self):
        return len(self.tuples)


class DS_parent:
    def __init__(self):
        group_ids = [0] # group_ids not used ATM
        self.group_ids = group_ids
        relative_batch_size = 1 # relative_batch_size not effective ATM
        self.batch_size = relative_batch_size
# group_id determines sampling scheme - one group_id is chosen randomly for each batch, single dataset may be in more group_id
# then the relative_batch_size (any positive number - applies as a ratio) determines how many patches are chosen from each dataset for inidividual batch
# each batch has size args.batch_size (may differ slightly if args.batch_size is not divisible by sum of relative sizes)


class DS_Brown(DS_parent):
    def __init__(self, path: str, fliprot: bool, transform_dict: dict):
        super().__init__()
        # group_ids = [0] # group_ids not used ATM
        # self.group_ids = group_ids
        # relative_batch_size = 1 # relative_batch_size not effective ATM
        self.path = path
        # self.batch_size = relative_batch_size
        self.fliprot = fliprot
        self.transform_dict = transform_dict

        bcolors.p(bcolors.YELLOW, str(self.__dict__))

class DS_AMOS(DS_parent):
    def __init__(self, tower_dataset, split_name, n_patch_sets, weight_function, fliprot, transform, patch_gen, cams_in_batch, masks_dir=None):
        super().__init__()
        # group_ids = [0] # group_ids not used ATM
        # self.group_ids = group_ids
        # relative_batch_size = 1 # relative_batch_size not effective ATM
        self.tower_dataset = tower_dataset
        self.split_name = split_name
        self.n_patch_sets = n_patch_sets
        self.weight_function = weight_function
        # self.batch_size = relative_batch_size
        self.fliprot = fliprot
        self.transform = transform
        self.patch_gen = patch_gen
        self.cams_in_batch = cams_in_batch
        self.masks_dir = masks_dir

        bcolors.p(bcolors.YELLOW, str(self.__dict__))


class DS_wrapper:
    def prepare_epoch(self, gen_tuples=True):
        kwargs = {"num_workers": max(1, 8//len(self.datasets)), "pin_memory": True}
        self.loaders = []
        label_offset = 0
        for c in self.datasets:
            if isinstance(c, DS_Brown):
                self.loaders += [
                    TotalDatasetsLoader(
                        train=True,
                        load_random_triplets=False,
                        batch_size=c.batch_size,
                        dataset_path=os.path.abspath(c.path),
                        fliprot=c.fliprot,
                        n_tuples=self.n_tuples,
                        transform_dict=c.transform_dict,
                        group_ids=c.group_ids,
                        label_offset=label_offset,
                    )
                ]
            elif isinstance(c, DS_AMOS):
                self.loaders += [
                    WBSDataset(
                        root=c.tower_dataset,
                        split_name=c.split_name,
                        n_patch_sets=c.n_patch_sets,
                        masks_dir=c.masks_dir,
                        weight_function=c.weight_function,
                        grayscale=True,
                        download=False,
                        group_ids=c.group_ids,
                        n_tuples=self.n_tuples,
                        batch_size=c.batch_size,
                        fliprot=c.fliprot,
                        transform=c.transform,
                        cams_in_batch=c.cams_in_batch,
                        patch_gen=c.patch_gen,
                        label_offset=label_offset,
                    )
                ]
            else:
                raise Exception("invalid DS class")
            label_offset = self.loaders[-1].max_label()+1

        for loader in self.loaders:
            loader.bs = {}

        for gid in self.all_group_ids:
            cur_loaders = [c for c in self.loaders if gid in c.group_ids]
            sum_of_sizes = sum([c.batch_size for c in cur_loaders])
            for loader in cur_loaders:
                loader.bs[gid] = int((loader.batch_size / sum_of_sizes) * self.batch_size)

        if not gen_tuples:
            return

        self.gid_seq = np.random.choice(self.all_group_ids, size=self.n_iters(), p=self.frequencies) # prepare sequence of groups for the epoch
        # print(self.gid_seq)
        for loader in self.loaders: # in each loader
            loader.bs_seq = [loader.bs[c] for c in self.gid_seq if c in loader.bs.keys()]
            loader.generate_tuples()

        self.gid_to_iters = {}
        for gid in self.all_group_ids:
            cur_loaders = [c for c in self.loaders if gid in c.group_ids]
            self.gid_to_iters[gid] = [iter(torch.utils.data.DataLoader(c, batch_size=c.bs[gid], shuffle=False, **kwargs)) for c in cur_loaders]

    def __init__(self, datasets: [DS_parent], n_tuples, batch_size, frequencies, fliprot=False):
        self.n_tuples = n_tuples
        self.gid_to_DS = {}
        self.b_size = batch_size
        self.datasets = datasets
        self.fliprot = fliprot
        self.batch_size = batch_size

        if not len(frequencies)==len(self.datasets):
            raise Exception('must be len(frequencies)==len(datasets), that is one relative frequency for each DS')
        self.frequencies = np.array(frequencies) / np.sum(np.array(frequencies))
        for i,d in enumerate(self.datasets):
            d.group_ids = [i]
        self.all_group_ids = list(set().union(*[c.group_ids for c in datasets]))

        self.prepare_epoch(gen_tuples=False)
        for gid in self.all_group_ids:
            print("group {} b_sizes: {}".format(gid, [z.bs[gid] for z in [c for c in self.loaders if gid in c.group_ids]]))

    def n_iters(self):
        return int(self.n_tuples / self.batch_size)

    def __getitem__(self, idx):
        if idx >= len(self.gid_seq):
            raise StopIteration
        gid = self.gid_seq[idx]
        data_a = None
        data_p = None
        for loader in self.gid_to_iters[gid]:
            pom_a, pom_p = next(loader)
            # print(pom_a.shape)
            if data_a is None:
                data_a = pom_a.float()
                data_p = pom_p.float()
            else:
                data_a = torch.cat((data_a.float(), pom_a.float()))
                data_p = torch.cat((data_p.float(), pom_p.float()))
        return data_a, data_p

    def __len__(self):
        return len(self.gid_seq)



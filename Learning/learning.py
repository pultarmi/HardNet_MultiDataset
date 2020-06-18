import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import kornia
# import albumentations
from Learning.losses import *
from tqdm.auto import tqdm
from Learning.eval_metrics import *
from utils_ import printc


def adjust_learning_rate(optimizer, orig_lr, batch_size, n_triplets, epochs):
    out = 0
    for group in optimizer.param_groups:
        if "no_grad" in group.keys():
            continue
        if "step" not in group:
            group["step"] = 0.0
        else:
            group["step"] += 1.0
        group["lr"] = orig_lr * (1.0 - float(group["step"]) * float(batch_size) / (n_triplets * float(epochs)))
        out = group["lr"]
    return out

def get_lr(optimizer):
    for group in optimizer.param_groups:
        if "no_grad" in group.keys():
            continue
        return group["lr"]

def null_steps(optimizer):
    for group in optimizer.param_groups:
        if "no_grad" in group.keys():
            continue
        group["step"] = 0.0
    return optimizer

def create_optimizer(model, new_lr, optimizer_name, wd): # , train_loader=None
    # if optimizer_name == "sgd":
    #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                           lr=new_lr,
    #                           momentum=0.9,
    #                           dampening=0.9,
    #                           weight_decay=wd)
    # sigmas = []
    # for l in train_loader.sigmas.items():
    #     sigmas += [l[1]]
    if optimizer_name == "sgd":
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters())+sigmas ),
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters()) ),
                              lr=new_lr,
                              momentum=0.9,
                              dampening=0.9,
                              weight_decay=wd)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=new_lr, weight_decay=wd)
    else:
        raise Exception("Not supported optimizer: {0}".format(optimizer_name))
    return optimizer

def create_optimizer_LossNet(model, new_lr, optimizer_name, wd):
    if optimizer_name == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters() ),
                              lr=new_lr,
                              momentum=0.9,
                              dampening=0.9,
                              weight_decay=wd)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=new_lr, weight_decay=wd)
    else:
        raise Exception("Not supported optimizer: {0}".format(optimizer_name))
    return optimizer

def my_collate_fn(batch, key=''): # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    if type(elem) == type(dict()):
        return {key: my_collate_fn([d[key] for d in batch], key) for key in elem}
    if key in ['set_idxs']:
        return batch
    if key in ['data', 'labels']:
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        # print(batch[0].shape)
        return torch.cat(batch, 0, out=out)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    # if type(elem) == type([]):
    #     return batch

def test(loader, model, name):
    model.eval()
    with torch.no_grad():
        labels, distances = [], []
        pbar = tqdm(enumerate(loader), total=len(loader), desc=name + ' test')
        for batch_idx, (data_a, data_p, label) in pbar:
            out_a = model(data_a.cuda())
            out_p = model(data_p.cuda())
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy().reshape(-1, 1))
            labels.append(label.data.cpu().numpy().reshape(-1, 1))
        num_tests = loader.dataset.matches.size(0)
        labels = np.vstack(labels).reshape(num_tests)
        distances = np.vstack(distances).reshape(num_tests)
        fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
        printc.red('FPR95= {:.2f} AP= {:.2f}'.format(100 * fpr95, 100 * AP(labels, distances)))

def safe_transform(img, transformation=None):
    if transformation is not None:
        return transformation(img)
    # return img
    return img.unsqueeze(0) # because pytorch transform adds dimension


trans_resize32 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor()
])

trans_resize48 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(48),
    transforms.ToTensor()
])

trans_none = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

trans_crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

trans_crop_resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(64),
    transforms.Resize(32),
    transforms.ToTensor()
])

trans_crop_resize48 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(64),
    transforms.Resize(48),
    transforms.ToTensor()
])

trans_crop_resize64 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

transform_AMOS_s = transforms.Compose([ # CPU
    transforms.ToPILImage(),
    transforms.Pad(10, padding_mode='reflect'), # otherwise small black corners appear
    transforms.RandomAffine(25, scale=(0.8, 1.4), shear=25, resample=Image.BICUBIC),
    # transforms.CenterCrop(64),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

trans_AMOS = transforms.Compose([ # CPU
    transforms.ToPILImage(),
    transforms.Pad(10, padding_mode='reflect'), # otherwise small black corners appear
    transforms.RandomAffine(25, scale=(0.8, 1.4), shear=25, resample=Image.BICUBIC),
    transforms.CenterCrop(64),
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.9, 1.10)),
    transforms.ToTensor()
])


def get_transform_AMOS_kornia1(in_size=96, out_size=32):
    print(get_transform_AMOS_kornia1.__name__)
    return nn.Sequential(  # GPU
        kornia.filters.GaussianBlur2d((11, 11), (in_size / (2. * out_size), in_size / (2. * out_size))),
        torch.nn.ReplicationPad2d(in_size // 4),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(-15.0, 15.0), scale=(0.8, 1.2), shear=(-10, 10), translate=(0.0, 0.05)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size)),
    ).cuda()

def get_transform_AMOS_kornia2(in_size=96, out_size=32):
    print(get_transform_AMOS_kornia2.__name__)
    return nn.Sequential(  # GPU
        kornia.filters.GaussianBlur2d((11, 11), (in_size / (2. * out_size), in_size / (2. * out_size))),
        torch.nn.ReplicationPad2d(in_size // 4),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(-20.0, 20.0), scale=(0.8, 1.1), shear=(-15, 15), translate=(0.0, 0.05)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size)),
    ).cuda()

def get_transform_AMOS_kornia3(in_size=96, out_size=32):
    print(get_transform_AMOS_kornia3.__name__)
    return nn.Sequential(  # GPU
        kornia.filters.GaussianBlur2d((11, 11), (in_size / (2. * out_size), in_size / (2. * out_size))),
        torch.nn.ReplicationPad2d(in_size // 4),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(-25.0, 25.0), scale=(0.7, 1.1), shear=(-20, 20), translate=(0.0, 0.1)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size)),
    ).cuda()

def get_transform_AMOS_kornia4(in_size=96, out_size=32):
    print(get_transform_AMOS_kornia4.__name__)
    return nn.Sequential(  # GPU
        # kornia.filters.GaussianBlur2d((11, 11), (orig_size / (2. * out_size), orig_size / (2. * out_size))),
        torch.nn.ReplicationPad2d(in_size // 4),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(-25.0, 25.0), scale=(0.7, 1.1), shear=(-20, 20), translate=(0.0, 0.1)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size)),
    ).cuda()

def get_transform_lib_kornia(in_size=64):
    print(get_transform_lib_kornia.__name__)
    out_size = 32
    return nn.Sequential(  # GPU
        kornia.filters.GaussianBlur2d((11, 11), (in_size / (2. * out_size), in_size / (2. * out_size))),
        torch.nn.ReplicationPad2d(in_size // 4),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(-15.0, 15.0), scale=(0.8, 1.2), shear=(-10, 10), translate=(0.0, 0.07)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size)),
    ).cuda()

def get_transform_lib_kornia_noblur(in_size=64):  # best for liberty
    print(get_transform_lib_kornia2.__name__)
    out_size = 32
    return nn.Sequential(
        # kornia.filters.GaussianBlur2d((7, 7), (0.6, 0.6)),  # Blur for proper downscale
        torch.nn.ReplicationPad2d(8),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(-5.0, 5.0), scale=(0.9, 1.0), shear=(5.0, 5.0), translate=(0.03, 0.03)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size))
    ).cuda()

def get_transform_lib_kornia2(in_size=64):  # best for liberty
    print(get_transform_lib_kornia2.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.filters.GaussianBlur2d((7, 7), (0.6, 0.6)),  # Blur for proper downscale
        torch.nn.ReplicationPad2d(8),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(-5.0, 5.0), scale=(0.9, 1.0), shear=(5.0, 5.0), translate=(0.03, 0.03)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size))
    ).cuda()

def get_transform_lib_32(in_size=64):  # best for liberty
    print(get_transform_lib_32.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.Resize((out_size, out_size))
    ).cuda()

def get_transform_lib_48(in_size=64):  # best for liberty
    print(get_transform_lib_48.__name__)
    out_size = 48
    return nn.Sequential(
        kornia.Resize((out_size, out_size))
    ).cuda()

def get_transform_lib_64(in_size=64):  # best for liberty
    print(get_transform_lib_64.__name__)
    out_size = 64
    return nn.Sequential(
        kornia.Resize((out_size, out_size))
    ).cuda()

def get_transform_lib_kornia2s(in_size=64):  # milder
    print(get_transform_lib_kornia2s.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.filters.GaussianBlur2d((7, 7), (0.6, 0.6)),  # Blur for proper downscale
        torch.nn.ReplicationPad2d(8),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(-5.0, 5.0), scale=(0.9, 1.0), shear=(5.0, 5.0), translate=(0.00, 0.00)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size))
    ).cuda()

def get_transform_lib_kornia2ss(in_size=64):  # milder
    print(get_transform_lib_kornia2ss.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.filters.GaussianBlur2d((7, 7), (0.6, 0.6)),  # Blur for proper downscale
        torch.nn.ReplicationPad2d(8),  # otherwise small black corners appear
        kornia.augmentation.RandomAffine(degrees=(0.0, 0.0), scale=(1.0, 1.0), shear=(5.0, 5.0), translate=(0.00, 0.00)),
        kornia.augmentation.CenterCrop(in_size),
        kornia.Resize((out_size, out_size))
    ).cuda()

def get_transform_lib_kornia_onlyrot(in_size=64):
    print(get_transform_lib_kornia_onlyrot.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.augmentation.RandomRotation(degrees=(-5.0, 5.0)),
        kornia.augmentation.CenterCrop(out_size),
    ).cuda()

def get_transform_lib_kornia_onlyshift(in_size=64):
    print(get_transform_lib_kornia_onlyshift.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.augmentation.RandomAffine(degrees=(0.0, 0.0), translate=(0.03, 0.03)),
        kornia.augmentation.CenterCrop(out_size),
    ).cuda()

def get_transform_lib_kornia_resize32(in_size=64):
    print(get_transform_lib_kornia_resize32.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.Resize((out_size, out_size)),
    ).cuda()

def get_transform_lib_kornia_resize_crop(in_size=64):
    print(get_transform_lib_kornia_resize_crop.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.Resize((48, 48)),
        kornia.augmentation.CenterCrop((out_size, out_size)),
    ).cuda()

def get_transform_lib_kornia_crop(in_size=64):
    print(get_transform_lib_kornia_crop.__name__)
    out_size = 32
    return nn.Sequential(
        kornia.augmentation.CenterCrop(out_size),
    ).cuda()

class KoT:
    which_T = ''

    @staticmethod
    def set_kornia_tr(name):
        KoT.which_T = name

    all_kornia_transforms = \
        {
            '': None,
            'full': { # best for liberty
                'AMOS': get_transform_AMOS_kornia4,
                'other': get_transform_lib_kornia2,
            },
            'resize32': {
                'other': get_transform_lib_32,
            },
            'resize48': {
                'other': get_transform_lib_48,
            },
            'resize64': {
                'other': get_transform_lib_64,
            },
            'resize_crop': { # for test of flatness
                'other': get_transform_lib_kornia_resize_crop,
            },
            'noblur': {
                'other': get_transform_lib_kornia_noblur,
            },
        }

    @staticmethod
    def get_transform_kornia():
        aux = KoT.all_kornia_transforms[KoT.which_T]
        if aux is not None:
            aux = {k:aux[k]() for k in aux.keys()} # initialize only the returned ones
        return aux


# def transform_AMOS_alb(img): # CPU
#     img = img.permute(2,1,0).data.cpu().numpy()
#     f = albumentations.Compose([
#         albumentations.PadIfNeeded(img.shape[1]+20,img.shape[1]+20, border_mode=cv.BORDER_REFLECT),
#         albumentations.ShiftScaleRotate(rotate_limit=25, scale_limit=(0.8, 1.4), interpolation=cv.INTER_LINEAR),
#         albumentations.CenterCrop(64,64, always_apply=True),
#         albumentations.RandomResizedCrop(32,32, scale=(0.7, 1.0), ratio=(0.9, 1.10)),
#     ])
#     res = f(image=img)
#     res = torch.tensor(res['image'])
#     res = res.permute(2,1,0)
#     return res
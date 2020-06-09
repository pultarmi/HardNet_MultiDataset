import cv2, argparse
import torch.nn.init
from tqdm import tqdm
from utils_ import *
from architectures import *
# from torch.utils.serialization import load_lua
from glob import glob

parser = argparse.ArgumentParser(description="HPatches descs extractor")
parser.add_argument("--model_name", default='')
parser.add_argument("--hpatches_dir", default='/mnt/datagrid/personal/qqpultar/hpatches-benchmark/data/hpatches-release')
parser.add_argument("--output_dir", default='HP_descs')
parser.add_argument("--bs", type=int, default=500)
parser.add_argument("--overwrite", default=False, action="store_true", help="overwrite existing files")
args = parser.parse_args()
printc.yellow("parsed options:\n{}\n".format(vars(args)))

tps = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5', 't1', 't2', 't3', 't4', 't5'] # all types of patches

class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self, base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t + '.png')
            im = cv2.imread(im_path, 0)
            self.N = im.shape[0] / 65
            setattr(self, t, np.split(im, self.N))

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x

w = 65
ps = 32
# ps = 48
# ps = 64
bs = args.bs

model = load_hardnet(args.model_name)
desc_name = 'HardNet_' + args.model_name
model.eval()
try:
    ps = model.isize
    print('ps set to', ps)
except:
    pass

seqs = [os.path.abspath(p) for p in glob(args.hpatches_dir + '/*')]
printc.green('found {} seqs'.format(len(seqs)))
pbar = tqdm(seqs)
for seq_path in pbar:
    seq = hpatches_sequence(seq_path)
    path = os.path.join(args.output_dir, os.path.join(desc_name, seq.name))
    os.makedirs(path, exist_ok=True)
    for tp in tps:
        pbar.set_description(seq.name)
        if os.path.isfile(os.path.join(path, tp + '.csv')) and not args.overwrite:
            continue
        n_patches = len(getattr(seq, tp))
        patches_for_net = np.zeros((n_patches, 1, ps, ps))
        for i, patch in enumerate(getattr(seq, tp)):
            patches_for_net[i, 0] = cv2.resize(patch[0:w, 0:w], (ps, ps))

        one_descs = []
        n_batches = int(n_patches / bs + 1)
        for batch_idx in range(n_batches):
            st = batch_idx * bs
            if (batch_idx == n_batches - 1) and ((batch_idx + 1) * bs > n_patches):
                end = n_patches
            else:
                end = (batch_idx + 1) * bs
            if st >= end:
                continue
            data_a = patches_for_net[st:end].astype(np.float32)
            data_a = torch.from_numpy(data_a).cuda().detach()
            with torch.no_grad():
                out_a = model(data_a)
            one_descs.append(out_a.data.cpu().numpy())
        descs = np.concatenate(one_descs)
        np.savetxt(os.path.join(path, tp + '.csv'), descs, delimiter=',', fmt='%10.5f')
print('DONE')
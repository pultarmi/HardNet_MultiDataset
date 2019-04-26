"""
source (with minor changes): https://raw.githubusercontent.com/DagnyT/hardnet/master/code/dataloaders/HPatchesDatasetCreator.py
credit goes to the author Anastasiia Mishchuk
"""
import os, cv2, sys, json, torch
import numpy as np
import torch.utils.data as data

types = ['e1','e2','e3','e4','e5','ref','h1','h2','h3','h4','h5', 't1', 't2', 't3', 't4', 't5']

def mean_image(patches):
    mean = np.mean(patches)
    return mean

def std_image(patches):
    std = np.std(patches)
    return std

class HPatches(data.Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform

    def read_image_file(self, data_dir, exclude={'v_bark', 'v_boat', 'v_graffiti', 'v_wall'}): # exclude some scenes suspected to be in CAIP2019 test set
        """Return a Tensor containing the patches"""
        print('splits:',splits)
        patches, labels, counter = [], [], 0
        txts = []
        hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        for directory in hpatches_sequences:
            fn = os.path.basename(directory)
            if sum([c[0]+'_' in fn for c in splits])==0:
                continue
            if fn in exclude:
                print('XXXXX', fn)
                continue
            print(directory)
            for type in types:
                sequence_path = os.path.join(data_dir, directory,type)+'.png'
                image = cv2.imread(sequence_path, 0)
                h, w = image.shape
                n_patches = int(h / w)
                for i in range(n_patches):
                    patch = image[i * (w): (i + 1) * (w), 0:w]
                    patch = np.array(cv2.resize(patch, (64, 64)), dtype=np.uint8)
                    patches += [patch]
                    labels += [i+counter]
                    txts += [type]
            counter += n_patches
        print(counter)
        return torch.ByteTensor(np.array(patches, dtype=np.uint8)), torch.LongTensor(labels), txts

if __name__ == '__main__':
    try:
        path_to_hpatches_dir = sys.argv[1]
        output_dir  = sys.argv[2]
        splits = [c for c in sys.argv[3:]]
    except:
        print("Wrong input format. Try python HPatchesDatasetCreator.py path_to_hpatches path_to_splits_json output_dir")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    t = 'train'
    hPatches = HPatches(good_fnames = None)
    res = hPatches.read_image_file(path_to_hpatches_dir)
    out_p = os.path.join(output_dir, 'hpatches_split_' + '-'.join(splits) +  '_' + t + '.pt')
    with open(out_p, 'wb') as f:
        torch.save(res, f)
    print('Saved', out_p)

# please run: python HPatchesDatasetCreator.py.py ../hpatches-release Datasets/
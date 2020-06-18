import torch.utils.checkpoint as cp
from torch.nn import Parameter
from torch.hub import load_state_dict_from_url
from fire import Fire
from torch.nn.utils import spectral_norm
# from fastai2.basics import *
from fastai2.vision.all import *
# from fastai2.callback.all import *
# from fastai2.distributed import *
# from fastai2.vision.models.xresnet import *
# from fastai2.callback.mixup import *
# from torch.utils import load_state_dict_from_url
from torch.nn import functional as tf
from Learning.learning import *
from sklearn.preprocessing import normalize


def load_hardnet(model_name):
    path = os.path.join('Models', model_name, model_name + '.pt')
    return load_hardnet_abs(path)

def load_hardnet_abs(model_path): # returns pretrained model from absolute path, use model.eval() before testing
    print('Loading HardNet ...')
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
        model = get_model_by_name(checkpoint['model_arch'])
        model.cuda()
    else:
        print('no gpu')
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = get_model_by_name(checkpoint['model_arch'])
    model.load_state_dict(checkpoint['state_dict'])
    if 'pca' in checkpoint.keys():
        print('LOADING PCA')
        model.pca = checkpoint['pca']
    return model

def obsolete_map(name):
    m = {'h1':'h7', 'h3':'h8'}
    if name in m.keys():
        print('USING OBSOLETE NAME')
        return m[name]
    return name

def get_model_by_name(name):
    nlist = name.replace('_','-').split('-')
    print('arch:', nlist)
    if len(nlist)==1: nlist += [1] # assume grayscale
    nlist[0] = obsolete_map(nlist[0])
    model = archs()[nlist[0]](channels=int(nlist[1]))
    model.name = name
    model.kornia_transform = KoT.get_transform_kornia()
    return model

def archs():
    return{
        'h7': HardNet,
        'h7d01': HardNetd01,
        'h7nodrop': HardNetdrop0,
        'h7x2': HardNetx2,
        'h7p48': HardNet_p48,
        'h7Le': HardNetLe,
        # 'h7x2': HardNet_7_256,
        'h8': HardNet_8_256,
        'h8Ins': HardNet_8_256_Ins,
        'h8c': HardNet_8_128,
        'h8p48': HardNet_8_256_p48,
        'h8p48x': HardNet_8_256_p48x,
        'h8p64': HardNet_8_256_p64,
        'h8avg': HardNet_8_256_avg,
        'h8max': HardNet_8_256_max,
        'h8x2': HardNet_8_512,
        'h8x2c': HardNet_8_512C, # less channel on begin
        'h8sa': HardNet_SA,
        'h8mb': HardNet_MB,

        'h8pca256': HardNet_8_256_pca256,
        'h8pca128': HardNet_8_256_pca128,
        'h8pca96': HardNet_8_256_pca96,
        'h8pca64': HardNet_8_256_pca64,
        'h8pca32': HardNet_8_256_pca32,

        'h7pca128': HardNetPCA128,
        'h7pca96': HardNetPCA96,
        'h7pca64': HardNetPCA64,
        'h7pca32': HardNetPCA32,

        # Exp 2 hardnet8_256, replace global pooling to
        'h8Econv3x3s2p1': HardNet_8_256Econv3x3_s2p1,
        'h8Econv3x3s2s2c22': HardNet_8_256Econv3x3_s2,
        'h8Econv3xnopad': HardNet_8_256Enopad,

        'h8blocks': HardNet_8_256_blocks,
        'h8blocksc3x3': HardNet_8_256_blocksc3x3,

        # exp 3
        'h8Eavg11': HardNet_8_256_avg11,
        'h8Emax11': HardNet_8_256_max11,
        # exp 4
        'h7E256': HardNet256,
        'h8Ecto128': HardNet_8_256_cto128,

        'h8E512': HardNet_8_256_512,
        'h8E512pca512': HardNet_8_256_512pca512,
        'h8E512pca256': HardNet_8_256_512pca256,
        'h8E512pca128': HardNet_8_256_512pca128,
        'h8E512pca96': HardNet_8_256_512pca96,
        'h8E512pca64': HardNet_8_256_512pca64,
        'h8E512pca32': HardNet_8_256_512pca32,

        'h9': HardNet_9_256,
        'h9cl': HardNet_9_256_convlater,
        'h9x': HardNet_9_256x,
        'h9t': HardNet_9_256t,
        'h9tt': HardNet_9_256tt,

        'HReM': HardNetReshapeMishRes,
        'HReMIn': HardNetReshapeMishConvIN,
        'h8SBMIn': HardNetSingleBlockMishConvIN,

        'HR': HardNetR,
        'HR2': HardNetR2,
        'hPS': HardNetPS,
        'sosnet': SOSNet32x32,

        'h7sep': HardNet_7_sep,  # channels=2
        'h8sep': HardNet_8_sep,  # channels=2
        'h8pre': HardNet_8_256_pre,  # channels=3
        'h8E512pre': HardNet_8_512_pre,  # channels=3
        'h8x2pre': HardNet_8_256l_pre,  # channels=3

        'De32': partial(DenseNet,growth_rate=32,block_config=(6, 12, 24, 16),bn_size=4, drop_rate=0.3, Nout_features=128,num_init_features=64),
        'De16': partial(DenseNet,growth_rate=16,block_config=(6, 12, 24, 16),bn_size=4, drop_rate=0.3, Nout_features=128,num_init_features=64),
        'De16s': partial(DenseNet,growth_rate=16,block_config=(6, 8, 16, 12),bn_size=4, drop_rate=0.3, Nout_features=128,num_init_features=64),
        'De16xs': partial(DenseNet,growth_rate=16,block_config=(6, 8, 16, 12),bn_size=4, drop_rate=0.3, Nout_features=128,num_init_features=32),

        # 'R1': partial(ResNet,channels=1,block=BasicBlock,layers=[3, 4, 6, 3]),
        # 'R2': partial(ResNet2,channels=1,block=BasicBlock,layers=[3, 4, 6, 3]),
        # 'R3': partial(ResNet3,channels=1,block=BasicBlock,layers=[3, 4, 6, 3]),

        # 'R4': partial(ResNet4,channels=1,layers=[3, 4, 6, 3]),
        # 'R4s': partial(ResNet4s,channels=1,layers=[3, 4, 6, 3]),
        # 'R4ss': partial(ResNet4ss,channels=1,layers=[3, 4, 6]),
        # 'R4sss': partial(ResNet4sss,channels=1,layers=[3, 4, 6]),

        # 'R5': ResNet5,
        # 'R6': ResNet6,
        # 'R6x': partial(ResNet6, layers=[32,64,64,128]),
        # 'R7': partial(ResNet7, layers=[32,64,128,256]),
        # 'R7x': partial(ResNet7, layers=[32,64,128,256]),

        # 'R5': ResNet_new,
        'Rc3': partial(ResNet_new, layers=[16,32,64,128], resblock=ResBlock_c3),
        'Rc3x': partial(ResNet_new, layers=[32, 64, 64, 128], resblock=ResBlock_c3),
        'Rc2': partial(ResNet_new, layers=[32, 64, 64, 128], resblock=ResBlock_c2),

        'l2net': L2Net,
    }

def try_kornia(x, transform):
    if (transform is not None) and (type(x) == type({})): # this format is only during training
        with torch.no_grad():
            return transform[x['loader']](x['data'])
    return x

def lossnet_from_name(name):
    if name in ['l1']:
        return LossNet()
    if name in ['l2']:
        return LossNet2()
    if name in ['l3']:
        return LossNet3()
    assert False, 'wrong model architecture, change --model_arch'

class BaseHardNet(nn.Module):
    def forward(self, x):
        x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        assert x.shape[1] == self.osize, 'osize!=number of channels, check model arch'
        return F.normalize(x,p=2)

class BaseHardNetPCA(nn.Module):
    def forward(self, x):  # cannot learn !!!
        x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        x = normalize(self.pca.transform(x.cpu().detach().numpy()))  ############
        x = torch.tensor(x)
        assert x.shape[1] == self.osize
        return x

class HardNet(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
        self.features.apply(weights_init)

class HardNetd01(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetd01, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetPCA96(BaseHardNetPCA):
    osize = 96
    def __init__(self, channels=1):
        super(HardNetPCA96, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
        self.features.apply(weights_init)

class HardNetPCA128(BaseHardNetPCA):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetPCA128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
        self.features.apply(weights_init)

class HardNetPCA64(BaseHardNetPCA):
    osize = 64
    def __init__(self, channels=1):
        super(HardNetPCA64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
        self.features.apply(weights_init)

class HardNetPCA32(BaseHardNetPCA):
    osize = 32
    def __init__(self, channels=1):
        super(HardNetPCA32, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
        self.features.apply(weights_init)

class HardNet256(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNetdrop0(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetdrop0, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            # nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetx2(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNetx2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_p48(BaseHardNet):
    isize = 48
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_p48, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 24
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 12
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), # 6
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=6, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetLe(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetLe, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetR(nn.Module):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetR, self).__init__()
        self.c1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(32, affine=False)
        # nn.ReLU(),
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(32, affine=False)
        # nn.ReLU(),
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(64, affine=False)
        # nn.ReLU(),
        self.c4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(64, affine=False)
        # nn.ReLU(),
        self.c5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.b5 = nn.BatchNorm2d(128, affine=False)
        # nn.ReLU(),
        self.c6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.b6 = nn.BatchNorm2d(128, affine=False)
        # nn.ReLU(),
        self.d = nn.Dropout(0.3)
        self.c7 = nn.Conv2d(128, 128, kernel_size=8, bias=False)
        self.b7 = nn.BatchNorm2d(128, affine=False)

        for m in self.modules():
            weights_init(m)

    def forward(self, input):
        x = norm_HW(input.float())

        x = tf.relu(self.b1(self.c1(x)))
        x = x + tf.relu(self.b2(self.c2(x)))
        x = tf.relu(self.b3(self.c3(x)))
        x = x + tf.relu(self.b4(self.c4(x)))
        x = tf.relu(self.b5(self.c5(x)))
        x = x + tf.relu(self.b6(self.c6(x)))
        x = self.d(x)
        x = tf.relu(self.b6(self.c6(x)))

        x = x.view(x.size(0), -1)
        return F.normalize(x,p=2)

class HardNetR2(nn.Module):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetR2, self).__init__()
        self.c1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(32, affine=False)
        # nn.ReLU(),
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(32, affine=False)

        self.c2a = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.b2a = nn.BatchNorm2d(64, affine=False)
        # nn.ReLU(),
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(64, affine=False)
        # nn.ReLU(),
        self.c4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(64, affine=False)

        self.c4a = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.b4a = nn.BatchNorm2d(128, affine=False)
        # nn.ReLU(),
        self.c5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.b5 = nn.BatchNorm2d(128, affine=False)
        # nn.ReLU(),
        self.c6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.b6 = nn.BatchNorm2d(128, affine=False)

        self.c6a = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.b6a = nn.BatchNorm2d(128, affine=False)
        # nn.ReLU(),
        self.d = nn.Dropout(0.3)
        self.c7 = nn.Conv2d(128, 128, kernel_size=8, bias=False)
        self.b7 = nn.BatchNorm2d(128, affine=False)

        for m in self.modules():
            weights_init(m)

    def forward(self, input):
        x = norm_HW(input.float())

        a = tf.relu(self.b1(self.c1(x)))
        x = tf.relu(x + self.b2(self.c2(a)))

        x = tf.relu(self.b2a(self.c2a(x)))

        a = tf.relu(self.b3(self.c3(x)))
        x = tf.relu(x + self.b4(self.c4(a)))

        x = tf.relu(self.b4a(self.c4a(x)))

        a = tf.relu(self.b5(self.c5(x)))
        x = tf.relu(x + self.b6(self.c6(a)))

        x = tf.relu(self.b6a(self.c6a(x)))

        x = self.d(x)
        x = tf.relu(self.b6(self.c6(x)))

        x = x.view(x.size(0), -1)
        return F.normalize(x,p=2)

class HardNet_7_sep(nn.Module): # gray + depth
    osize = 128
    def __init__(self, channels=None):
        super(HardNet_7_sep, self).__init__()
        self.features_g = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.ReLU())
        self.features_d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.ReLU())

        self.features_cat = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False)
        )

        self.features_g.apply(weights_init)
        self.features_d.apply(weights_init)
        self.features_cat.apply(weights_init)

    def forward(self, input):
        I0 = input[:,0,:,:].unsqueeze(1).float()
        I1 = input[:,1,:,:].unsqueeze(1).float()
        del input
        cat_features = self.features_cat(torch.cat([self.features_g(norm_HW(I0)), self.features_d(norm_HW(I1))], 1))
        x = cat_features.view(cat_features.size(0), -1)
        return F.normalize(x,p=2)

class HardNet_8_sep(nn.Module): # gray + depth
    osize = 128
    def __init__(self, channels=None):
        super(HardNet_8_sep, self).__init__()
        self.features_g = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.ReLU())
        self.features_d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.ReLU())

        self.features_cat = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False)
        )

        self.features_g.apply(weights_init)
        self.features_d.apply(weights_init)
        self.features_cat.apply(weights_init)

    def forward(self, input):
        I0 = input[:,0,:,:].unsqueeze(1).float()
        I1 = input[:,1,:,:].unsqueeze(1).float()
        del input
        cat_features = self.features_cat(torch.cat([self.features_g(norm_HW(I0)), self.features_d(norm_HW(I1))], 1))
        x = cat_features.view(cat_features.size(0), -1)
        return F.normalize(x,p=2)

class HardNet_7_256(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_7_256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_pre(BaseHardNet):
    osize = 256
    def __init__(self, channels=3):
        super(HardNet_8_256_pre, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 10, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(10, affine=False),
            nn.Tanh(),
            nn.Conv2d(10, 1, kernel_size=1, padding=0, bias=False),
            nn.Tanh(),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_512_pre(BaseHardNet):
    osize = 512
    def __init__(self, channels=3):
        super(HardNet_8_512_pre, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 10, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(10, affine=False),
            nn.Tanh(),
            nn.Conv2d(10, 1, kernel_size=1, padding=0, bias=False),
            nn.Tanh(),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256l_pre(BaseHardNet):
    osize = 256
    def __init__(self, channels=3):
        super(HardNet_8_256l_pre, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 10, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(10, affine=False),
            nn.Tanh(),
            nn.Conv2d(10, 1, kernel_size=1, padding=0, bias=False),
            nn.Tanh(),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256Econv3x3_s2p1(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256Econv3x3_s2p1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256Econv3x3_s2(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256Econv3x3_s2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256Enopad(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256Enopad, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_pca128(BaseHardNetPCA):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_8_256_pca128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

    # def forward(self, x): # cannot learn !!!
    #     x = try_kornia(x, self.kornia_transform)
    #     x = self.features(norm_HW(x.float()))
    #     x = x.view(x.size(0), -1)
    #     x = F.normalize(x,p=2)
    #     x = normalize(self.pca.transform(x.cpu().detach().numpy())) ############
    #     x = torch.tensor(x)
    #     assert x.shape[1] == self.osize
    #     return x

class HardNet_8_256_pca256(BaseHardNetPCA):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_pca256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_pca96(BaseHardNet):
    osize = 96
    def __init__(self, channels=1):
        super(HardNet_8_256_pca96, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, x):  # cannot learn !!!
        x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        x = normalize(self.pca.transform(x.cpu().detach().numpy()))  ############
        x = torch.tensor(x)
        assert x.shape[1] == self.osize
        return x

class HardNet_8_256_pca64(BaseHardNet):
    osize = 64
    def __init__(self, channels=1):
        super(HardNet_8_256_pca64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, x):  # cannot learn !!!
        x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        x = normalize(self.pca.transform(x.cpu().detach().numpy()))  ############
        x = torch.tensor(x)
        assert x.shape[1] == self.osize
        return x

class HardNet_8_256_pca32(BaseHardNet):
    osize = 32
    def __init__(self, channels=1):
        super(HardNet_8_256_pca32, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, x):  # cannot learn !!!
        x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        x = normalize(self.pca.transform(x.cpu().detach().numpy()))  ############
        x = torch.tensor(x)
        assert x.shape[1] == self.osize
        return x

class HardNet_8_256_512(BaseHardNet):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_8_256_512, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca256(BaseHardNetPCA):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca512(BaseHardNetPCA):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca512, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca128(BaseHardNetPCA):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca96(BaseHardNetPCA):
    osize = 96
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca96, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca64(BaseHardNetPCA):
    osize = 64
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca32(BaseHardNetPCA):
    osize = 32
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca32, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_cto128(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_8_256_cto128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_Ins(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_Ins, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_blocks(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_blocks, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 64, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.Flatten(),
            nn.BatchNorm1d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_blocksc3x3(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_blocksc3x3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.Flatten(),
            nn.BatchNorm1d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_128(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_8_128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_p48(BaseHardNet):
    isize = 48
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_p48, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 24
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 12
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 6
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=6, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_p48x(BaseHardNet):
    isize = 48
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_p48x, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 24
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 12
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=12, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_p64(BaseHardNet):
    isize = 64
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_p64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 32
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 16
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 8
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_avg(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_avg, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv2d(256, 256, kernel_size=8, bias=False),
            # nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_avg11(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_avg11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.features.apply(weights_init)

class HardNet_8_256_max(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_max, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=8, stride=1, padding=0),
            # nn.Conv2d(256, 256, kernel_size=8, bias=False),
            # nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_max11(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_max11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.MaxPool2d(kernel_size=8, stride=1, padding=0),
        )
        self.features.apply(weights_init)

def _conv1d_spect(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)

class SimpleSelfAttention(Module):
    def __init__(self, n_in:int, ks=1):
        self.bn = nn.BatchNorm1d(n_in)
        self.bn.weight.data.fill_(0.1)
        self.bn.bias.data.fill_(0)
        self.conv = _conv1d_spect(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self,x):
        size = x.size()
        x = x.view(*size[:2],-1)
        xbn = self.bn(x)
        convx = self.conv(xbn)
        xxT = torch.bmm(xbn,xbn.permute(0,2,1).contiguous()).clamp_(-10,10)
        o = torch.bmm(xxT, convx.view(*size[:2],-1))
        o = F.tanh(self.gamma) * o + x
        return o.view(*size).contiguous()

class HardNet_SA(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_SA, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            SimpleSelfAttention(128),
            nn.ReLU(),
            nn.BatchNorm2d(128, affine=False),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_MB(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_MB, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            kornia.contrib.MaxBlurPool2d(3),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            kornia.contrib.MaxBlurPool2d(3),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_512(BaseHardNet):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_8_512, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_9_256(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_9_256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_9_256_convlater(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_9_256_convlater, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_9_256t(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_9_256t, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_9_256tt(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_9_256tt, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_9_256x(BaseHardNet):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_9_256x, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_512C(BaseHardNet):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_8_512C, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNetPS(nn.Module):
    osize = 128
    def __init__(self, channels=None):
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
            nn.Conv2d(128, 128, kernel_size=8, bias=True),
        )

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return F.normalize(x, p=2)


class SOSNet32x32(nn.Module):
    osize = 128
    """
    128-dimensional SOSNet model definition trained on 32x32 patches
    """
    def __init__(self, dim_desc=128, drop_rate=0.1, channels=None):
        super(SOSNet32x32, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        eps_fea_norm = 1e-5

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,

            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, patch, eps_l2_norm = 1e-10):
        descr = self.desc_norm(self.layers(patch) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr

class L2Net(nn.Module):
    """L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space"""
    def __init__(self, out_dim=128, binary=False, dropout_rate=0.1, channels=None):
        super().__init__()
        self._binary = binary
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, out_dim, kernel_size=8, bias=False),
            nn.BatchNorm2d(out_dim, affine=False),
        )

        if self._binary:
            self.binarizer = nn.Tanh()
        self.features.apply(L2Net.weights_init)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        input = self.input_norm(input)
        x = self.features(input)
        x = x.view(x.size(0), -1)
        if self._binary:
            return self.binarizer(x)
        else:
            return F.normalize(x, p=2, dim=1)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1, bias=True),
        )
        self.features.apply(weights_init)

    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        return x

class LossNet2(nn.Module):
    def __init__(self):
        super(LossNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1, bias=True),
        )
        self.features.apply(weights_init)

    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        return x

class LossNet3(nn.Module):
    def __init__(self):
        super(LossNet3, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1, bias=True),
        )
        self.features.apply(weights_init)

    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        return x


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)
        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_"""
    def __init__(self, channels=1, name='unset', growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, drop_rate=0.3, Nout_features=128, memory_efficient=True, num_init_features=64, load=True):
        super(DenseNet, self).__init__()
        self.name=name
        self.osize = Nout_features
        self.features = nn.Sequential(OrderedDict([ # First convolution
            ('conv01', nn.Conv2d(channels, num_init_features//4, kernel_size=3, stride=1, padding=1, bias=False)),
            ('relu01', nn.ReLU()),
            ('norm01', nn.BatchNorm2d(num_init_features//4)),
            ('conv02', nn.Conv2d(num_init_features//4, num_init_features//2, kernel_size=3, stride=1, padding=1, bias=False)),
            ('relu02', nn.ReLU()),
            ('norm02', nn.BatchNorm2d(num_init_features//2)),
            ('conv03', nn.Conv2d(num_init_features//2, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('relu04', nn.ReLU()),
            ('norm04', nn.BatchNorm2d(num_init_features)),

            # ('conv02', nn.Conv2d(channels, num_init_features // 2, kernel_size=3, stride=1,padding=1, bias=False)),
            # ('relu02', nn.ReLU(inplace=True)),
            # ('norm02', nn.BatchNorm2d(num_init_features // 2)),
            # ('conv03', nn.Conv2d(num_init_features // 2, num_init_features, kernel_size=3, stride=2,padding=1, bias=False)),
            # ('relu04', nn.ReLU(inplace=True)),
            # ('norm04', nn.BatchNorm2d(num_init_features)),

            # ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
            #                     padding=3, bias=False)),
            # ('norm0', nn.BatchNorm2d(num_init_features)),
            # ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features //= 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features)) # Final batch norm
        self.lastconv = nn.Conv2d(num_features, Nout_features, kernel_size=2, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features) # Final batch norm

        for m in self.modules(): # Official init from torch repo.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        if load:
            self._load_state_dict(model_urls['densenet121'], progress=True)

    def forward(self, x):
        features = self.features(norm_HW(x.float()))
        x = F.relu(features, inplace=True)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # out = torch.flatten(out, 1)
        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        # return out
        # x = out.view(out.size(0), -1)
        return F.normalize(x, p=2)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        loaded, failed = 0, 0
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter): # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
                loaded += 1
            except:
                failed += 1
        print('loaded',loaded,'failed',failed)

    def _load_state_dict(self, model_url, progress):
        printc.yellow('(pre)LOADING DNet WEIGHTS - whatever I can')
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used to find such keys.
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        state_dict = load_state_dict_from_url(model_url, progress=progress)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.load_my_state_dict(state_dict)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1): #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1): #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    # __constants__ = ['downsample']
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, channels, block, layers, zero_init_residual=False, name='unset', groups=1, width_per_group=64, Nout_features=128):
        super(ResNet, self).__init__()
        self.osize = Nout_features
        self.name = name

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lastconv = nn.Conv2d(512, Nout_features, kernel_size=1, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features) # Final batch norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(norm_HW(x.float()))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2)

class ResNet2(nn.Module):
    def __init__(self, channels, block, layers, zero_init_residual=False, name='unset', groups=1, width_per_group=64, Nout_features=128):
        super(ResNet2, self).__init__()
        self.osize = Nout_features
        self.name = name

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.middleconv = nn.Conv2d(512, Nout_features, kernel_size=1, stride=1, bias=False)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.lastconv = nn.Conv2d(512, Nout_features, kernel_size=1, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features) # Final batch norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(norm_HW(x.float()))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2)

class ResNet3(nn.Module):
    def __init__(self, channels, block, layers, zero_init_residual=False, name='unset', groups=1, width_per_group=64, Nout_features=128):
        super(ResNet3, self).__init__()
        self.osize = Nout_features
        self.name = name

        self.inplanes = 64
        # self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.middleconv = nn.Sequential(nn.Conv2d(512, Nout_features, kernel_size=4, stride=1, bias=False),
        #                                 nn.BatchNorm2d(Nout_features, affine=False))
        self.lastconv = nn.Conv2d(512, Nout_features, kernel_size=4, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features, affine=False) # Final batch norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, (nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm2d(planes * block.expansion, affine=False))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(norm_HW(x.float()))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2)

class ResNet4(nn.Module):
    def __init__(self, channels, layers, zero_init_residual=False, block=BasicBlock, name='unset', Nout_features=128):
        super(ResNet4, self).__init__()
        self.osize = Nout_features
        self.name = name

        self.inplanes = 64
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.lastconv = nn.Conv2d(512, Nout_features, kernel_size=4, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features, affine=False) # Final batch norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, (nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm2d(planes * block.expansion, affine=False))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(norm_HW(x.float()))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2)

class ResNet4s(nn.Module):
    def __init__(self, channels, layers, zero_init_residual=False, block=BasicBlock, name='unset', Nout_features=128):
        super(ResNet4s, self).__init__()
        self.osize = Nout_features
        self.name = name

        self.inplanes = 64
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.lastconv = nn.Conv2d(512, Nout_features, kernel_size=1, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features, affine=False) # Final batch norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, (nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm2d(planes * block.expansion, affine=False))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(norm_HW(x.float()))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2)

class ResNet4ss(nn.Module):
    def __init__(self, channels, layers, zero_init_residual=False, block=BasicBlock, name='unset', Nout_features=128):
        super(ResNet4ss, self).__init__()
        self.osize = Nout_features
        self.name = name

        self.inplanes = 64
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.lastconv = nn.Conv2d(256, Nout_features, kernel_size=4, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features, affine=False) # Final batch norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, (nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm2d(planes * block.expansion, affine=False))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(norm_HW(x.float()))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2)

class ResNet4sss(nn.Module):
    def __init__(self, channels, layers, zero_init_residual=False, block=BasicBlock, name='unset', Nout_features=128):
        super(ResNet4sss, self).__init__()
        self.osize = Nout_features
        self.name = name

        self.inplanes = 64
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.lastconv = nn.Conv2d(128, Nout_features, kernel_size=8, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features, affine=False) # Final batch norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, (nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm2d(planes * block.expansion, affine=False))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(norm_HW(x.float()))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # print(x.shape)
        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2)


class MaxBloorPool2dFast(nn.Module):
    # def __init__(self, ks = 2, border_type: str = 'reflect') -> None:
    def __init__(self, ks = 3, border_type: str = 'reflect') -> None:
        super(MaxBloorPool2dFast, self).__init__()
        self.border_type: str = border_type
        self.ks = ks
        self.blur = kornia.filters.GaussianBlur2d((7,7), (1.6, 1.6)).cuda()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        x_blur: torch.Tensor = self.blur(F.max_pool2d(input, self.ks, 1, padding=self.ks//2))
        out = F.avg_pool2d(x_blur, 1, 2)
        return out

class ResBlock_c2(nn.Module):
    # expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock_c2, self).__init__()
        self.block = nn.Sequential(
                nn.BatchNorm2d(inplanes, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,stride=1, bias = False),
                nn.BatchNorm2d(planes, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias = False),
                nn.BatchNorm2d(planes, affine=True),
        )
        with torch.no_grad():
            self.block[-1].weight.data *= 0.1
            self.block[-1].bias.data *= 0
        self.downsample = stride > 1
        if self.downsample:
            self.down = MaxBloorPool2dFast()
        if inplanes != planes:
            self.inc = nn.Conv2d(inplanes, planes, kernel_size=1)
        else:
            self.inc = nn.Sequential()
    def forward(self, x):
        if self.downsample:
            x1 = self.down(x)
        else:
            x1 = x
        identity = self.inc(x1)
        out = self.block(x1)
        out += identity
        return out


class ResNet5(nn.Module):
    osize = 512
    def __init__(self, channels=1, name=''):
        super(ResNet5, self).__init__()
        self.name = name
        resblock = ResBlock_c2
        self.features = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=False),
            nn.Conv2d(channels, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias = False),
            resblock(32,64, stride=2),
            resblock(64,64, stride=1),
            resblock(64,128, stride=2),
            resblock(128,128, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )

        self.features.apply(weights_init)

    def forward(self, x):
        x = try_kornia(x, self.kornia_transform)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # return L2Norm()(x)
        return F.normalize(x, p=2)


class ResBlock_c3(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock_c3, self).__init__()
        self.block = nn.Sequential(
                nn.BatchNorm2d(inplanes, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,stride=1, bias = False),
                nn.BatchNorm2d(planes, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias = False),
                nn.BatchNorm2d(planes, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias = False),
                nn.BatchNorm2d(planes, affine=True),
        )
        with torch.no_grad():
            self.block[-1].weight.data *= 0.1
            self.block[-1].bias.data *= 0
        self.downsample = stride > 1
        if self.downsample:
            self.down = MaxBloorPool2dFast()
        if inplanes != planes:
            self.inc = nn.Conv2d(inplanes, planes, kernel_size=1)
        else:
            self.inc = nn.Sequential()
    def forward(self, x):
        if self.downsample:
            x1 = self.down(x)
        else:
            x1 = x
        identity = self.inc(x1)
        out = self.block(x1)
        out += identity
        return out


class ResNet6(nn.Module):
    osize = 512
    def __init__(self, channels=1, name='', layers=[16,32,64,128], resblock = ResBlock_c3):
        super(ResNet6, self).__init__()
        self.name = name
        l = layers
        self.features = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=False),
            nn.Conv2d(channels, l[0], kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(l[0], affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(l[0], l[1], kernel_size=3, padding=1, bias = False),
            resblock(l[1],l[2], stride=2),
            resblock(l[2],l[2], stride=1),
            resblock(l[2],l[3], stride=2),
            resblock(l[3],l[3], stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(l[3], l[3], kernel_size=8, bias = False),
            nn.BatchNorm2d(l[3], affine=False),
        )

        self.features.apply(weights_init)

    def forward(self, x):
        x = try_kornia(x, self.kornia_transform)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # return L2Norm()(x)
        return F.normalize(x, p=2)

class ResNet7(nn.Module):
    def __init__(self, channels=1, name='', layers=[16,32,64,128]):
        super(ResNet7, self).__init__()
        self.name = name
        # resblock = ResBlock7
        resblock = ResBlock_c2
        l = layers
        self.features = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=False),
            nn.Conv2d(channels, l[0], kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(l[0], affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(l[0], l[1], kernel_size=3, padding=1, bias = False),
            resblock(l[1],l[1], stride=2),
            resblock(l[1],l[1], stride=1),
            resblock(l[1],l[2], stride=2),
            resblock(l[2],l[2], stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(l[2], l[2], kernel_size=8, bias = False),
            nn.BatchNorm2d(l[2], affine=False),
        )

        self.features.apply(weights_init)

    def forward(self, x):
        x = try_kornia(x, self.kornia_transform)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # return L2Norm()(x)
        return F.normalize(x, p=2)

class ResNet_new(nn.Module):
    osize = 128
    def __init__(self, channels=1, layers=[16,32,64,128], resblock = ResBlock_c2):
        super(ResNet_new, self).__init__()
        l = layers
        print('resblock', str(resblock))
        self.features = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=False),
            nn.Conv2d(channels, l[0], kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(l[0], affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(l[0], l[1], kernel_size=3, padding=1, bias = False),
            resblock(l[1],l[2], stride=2),
            resblock(l[2],l[2], stride=1),
            resblock(l[2],l[3], stride=2),
            resblock(l[3],l[3], stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(l[3], l[3], kernel_size=8, bias = False),
            nn.BatchNorm2d(l[3], affine=False),
        )

        self.features.apply(weights_init)

    def forward(self, x):
        x = try_kornia(x, self.kornia_transform)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # return L2Norm()(x)
        assert x.shape[1] == self.osize
        return F.normalize(x, p=2)


class SpaceToDepth2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 2, 2, W // 2, 2)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 4, H // 2, W // 2)  # (N, C*bs^2, H//bs, W//bs)
        return x

class HardNetReshapeMishRes(nn.Module):
    """HardNet model definition
    """
    osize = 256
    def __init__(self, channels=1):
        super(HardNetReshapeMishRes, self).__init__()
        self.block1 = nn.Sequential(
            SpaceToDepth2(),
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            Mish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            Mish(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            Mish(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            Mish(),
        )
        self.block1.apply(self.weights_init)
        self.block2 = nn.Sequential(
            SpaceToDepth2(),
            nn.Conv2d(4 * 64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            Mish(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            Mish(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            Mish(),
        )
        self.block2.apply(self.weights_init)

        self.global_pool1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(64, 256, kernel_size=16, bias=False),
            nn.BatchNorm2d(256, affine=False),
            Flatten()
        )
        self.global_pool1.apply(self.weights_init)
        self.global_pool2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
            Flatten()
        )

        self.global_pool2.apply(self.weights_init)

    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        return (x - mean) / (std + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            print(m.weight.sum())
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, input):
        x1 = self.block1(self.input_norm(input))
        x2 = self.block2(x1)
        x = self.global_pool2(x2) + self.global_pool1(x1)
        return F.normalize(x, p=2, dim=1)

class HardNetReshapeMishConvIN(nn.Module):
    """HardNet model definition"""
    osize=256
    def __init__(self, channels=1):
        super(HardNetReshapeMishConvIN, self).__init__()
        self.block1 = nn.Sequential(
            SpaceToDepth2(),
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            Mish(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            Mish(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            Mish(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            Mish(),
        )
        self.block1.apply(self.weights_init)
        self.block2 = nn.Sequential(
            SpaceToDepth2(),
            nn.Conv2d(4 * 128, 384, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(384, affine=False),
            Mish(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(384, affine=False),
            Mish(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(384, affine=False),
            Mish(),
        )
        self.block2.apply(self.weights_init)

        self.global_pool1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(128, 16, kernel_size=5, stride=4, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=False),
            Flatten(),

        )
        self.global_pool1.apply(self.weights_init)
        self.global_pool2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(384, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=False),
            nn.Flatten(),
            nn.BatchNorm1d(256, affine=False),

        )

        self.global_pool2.apply(self.weights_init)

    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        return (x - mean) / (std + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            print(m.weight.sum())
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, input):
        x1 = self.block1(self.input_norm(input))
        x2 = self.block2(x1)
        x = self.global_pool2(x2) + self.global_pool1(x1)
        return F.normalize(x, p=2, dim=1)

class HardNetSingleBlockMishConvIN(nn.Module):
    """HardNet model definition"""
    osize=256
    def __init__(self, channels=1):
        super(HardNetSingleBlockMishConvIN, self).__init__()
        self.block1 = nn.Sequential(
            SpaceToDepth2(),
            nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            Mish(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            Mish(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False),
            Mish(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=False),
            Mish(),
        )
        self.block1.apply(self.weights_init)

        self.global_pool1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(512, 16, kernel_size=5, stride=4, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=False),
            Flatten(),
            nn.BatchNorm1d(256, affine=False),

        )
        self.global_pool1.apply(self.weights_init)

    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        return (x - mean) / (std + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            print(m.weight.sum())
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, input):
        x1 = self.block1(self.input_norm(input))
        x = self.global_pool1(x1)
        return F.normalize(x, p=2, dim=1)


def norm_HW(x): # channel-wise mean, format is BCHW, dims 2,3 are H,W, broadcasting applies
    mp = torch.mean(x, dim=(2,3), keepdim=True)
    sp = torch.std(x, dim=(2,3), keepdim=True) + 1e-7
    return (x - mp) / sp

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except: pass

class Interface:
    def params(self, model=None):
        if model is not None:
            if type(model)==str:
                model = get_model_by_name(model)
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        res = []
        for a in archs().keys():
            aux = self.params(a)
            res += [(a,aux)]
            print(aux)
        print(' ≤ '.join([c[0] for c in sorted(res,key=lambda x:x[1])]))

if __name__ == "__main__":
    Fire(Interface)


# import torch
# import fastai2
# import sklearn
# import sklearn.decomposition.pca
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import normalize
#
# fname = 'h8E512libcoloPCA128lib.pt'
# w = torch.load(fname, map_location=lambda storage, loc: storage)
#
# import torch.nn as nn
# class TorchPCA(nn.Module):
#     def __init__(self, in_f, out_f):
#         super(TorchPCA, self).__init__()
#         self.linear = nn.Linear(in_f, out_f, bias = False)
#         self.mean = nn.Parameter(torch.zeros(in_f))
#     def init_from_sklearn(self, sklearn_pca):
#         with torch.no_grad():
#             self.linear.weight.data[:] = torch.from_numpy(sklearn_pca.components_).float()
#             self.mean.data[:] = torch.from_numpy(sklearn_pca.mean_).float()
#     def forward(self, input):
#         return self.linear(input - self.mean.unsqueeze(0))
#
# PCA_t = TorchPCA(512, 128)
# PCA_t.init_from_sklearn(w['pca'])
#
# aa = torch.rand(10, 512)
# sk_result = w['pca'].transform(aa.numpy())
# pt_result = PCA_t(aa).detach().numpy()
# print ((pt_result - sk_result).max())
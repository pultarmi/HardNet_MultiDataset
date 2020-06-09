import torch.nn as nn, scipy.ndimage
from Learning.LAF import abc2A, rectifyAffineTransformationUpIsUp, sc_y_x2LAFs
from utils_ import *
import torchvision.transforms as transforms
# from kornia import gaussian_blur
from PIL import Image, ImageDraw
import multiprocessing
import pathos


class ScalePyramid(nn.Module):
    def __init__(self, nLevels = 3, init_sigma = 1.6, border = 5):
        super(ScalePyramid,self).__init__()
        self.nLevels = nLevels
        self.init_sigma = init_sigma
        self.sigmaStep =  2 ** (1. / float(self.nLevels))
        #print 'step',self.sigmaStep
        self.b = border
        self.minSize = 2 * self.b + 2 + 1
        return
    def forward(self,x):
        pixelDistance = 1.0
        curSigma = 0.5
        if self.init_sigma > curSigma:
            sigma = np.sqrt(self.init_sigma**2 - curSigma**2)
            curSigma = self.init_sigma
            curr = GaussianBlur(sigma = sigma)(x)
        else:
            curr = x
        sigmas = [[curSigma]]
        pixel_dists = [[1.0]]
        pyr = [[curr]]
        j = 0
        while True:
            curr = pyr[-1][0]
            for i in range(1, self.nLevels + 2):
                sigma = curSigma * np.sqrt(self.sigmaStep*self.sigmaStep - 1.0 )
                #print 'blur sigma', sigma
                curr = GaussianBlur(sigma = sigma)(curr)
                curSigma *= self.sigmaStep
                pyr[j].append(curr)
                sigmas[j].append(curSigma)
                pixel_dists[j].append(pixelDistance)
                if i == self.nLevels:
                    nextOctaveFirstLevel = F.avg_pool2d(curr, kernel_size = 1, stride = 2, padding = 0)
            pixelDistance = pixelDistance * 2.0
            curSigma = self.init_sigma
            if (nextOctaveFirstLevel[0,0,:,:].size(0)  <= self.minSize) or (nextOctaveFirstLevel[0,0,:,:].size(1) <= self.minSize):
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append([curSigma])
            pixel_dists.append([pixelDistance])
            j+=1
        return pyr, sigmas, pixel_dists

class AffineShapeEstimator(nn.Module):
    def __init__(self, threshold=0.001, patch_size=19):
        super(AffineShapeEstimator, self).__init__()
        self.threshold = threshold
        self.PS = patch_size
        self.gx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[-1, 0, 1]]]], dtype=np.float32))
        self.gy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]]]], dtype=np.float32))
        self.gk = torch.from_numpy(CircularGaussKernel(kernlen=self.PS, sigma=(self.PS / 2) / 3.0).astype(np.float32))

    def invSqrt(self, a, b, c):
        eps = 1e-12
        mask = (b != 0).float()
        r1 = mask * (c - a) / (2.0 * b + eps)
        t1 = torch.sign(r1) / (torch.abs(r1) + torch.sqrt(1.0 + r1 * r1))
        r = 1.0 / torch.sqrt(1.0 + t1 * t1)
        t = t1 * r
        r = r * mask + 1.0 * (1.0 - mask)
        t = t * mask

        x = 1.0 / torch.sqrt(r * r * a - 2.0 * r * t * b + t * t * c)
        z = 1.0 / torch.sqrt(t * t * a + 2.0 * r * t * b + r * r * c)

        d = torch.sqrt(x * z)

        x = x / d
        z = z / d

        l1 = torch.max(x, z)
        l2 = torch.min(x, z)

        new_a = r * r * x + t * t * z
        new_b = -r * t * x + t * r * z
        new_c = t * t * x + r * r * z
        return new_a, new_b, new_c, l1, l2

    def forward(self, x):
        if x.is_cuda:
            self.gk = self.gk.cuda()
        else:
            self.gk = self.gk.cpu()
        gx = self.gx(F.pad(x, (1, 1, 0, 0), "replicate"))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), "replicate"))
        a1 = (gx * gx * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0), -1).mean(dim=1)
        b1 = (gx * gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0), -1).mean(dim=1)
        c1 = (gy * gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0), -1).mean(dim=1)
        a, b, c, l1, l2 = self.invSqrt(a1, b1, c1)
        rat1 = l1 / l2
        mask = (torch.abs(rat1) <= 6.0).float().view(-1)
        # return rectifyAffineTransformationUpIsUp(abc2A(a, b, c)), mask
        return rectifyAffineTransformationUpIsUp(abc2A(a, b, c))


class OrientationDetector(nn.Module):
    def __init__(self, mrSize=3.0, patch_size=None):
        super(OrientationDetector, self).__init__()
        if patch_size is None:
            patch_size = 32
        self.PS = patch_size
        self.bin_weight_kernel_size, self.bin_weight_stride = self.get_bin_weight_kernel_size_and_stride(self.PS, 1)
        self.mrSize = mrSize
        self.num_ang_bins = 36
        self.gx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))

        self.gy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))

        self.angular_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.angular_smooth.weight.data = torch.from_numpy(np.array([[[0.33, 0.34, 0.33]]], dtype=np.float32))

        self.gk = 10.0 * torch.from_numpy(CircularGaussKernel(kernlen=self.PS).astype(np.float32))

    def get_bin_weight_kernel_size_and_stride(self, patch_size, num_spatial_bins):
        bin_weight_stride = int(round(2.0 * np.floor(patch_size / 2) / float(num_spatial_bins + 1)))
        bin_weight_kernel_size = int(2 * bin_weight_stride - 1)
        return bin_weight_kernel_size, bin_weight_stride

    def get_rotation_matrix(self, angle_in_radians):
        angle_in_radians = angle_in_radians.view(-1, 1, 1)
        sin_a = torch.sin(angle_in_radians)
        cos_a = torch.cos(angle_in_radians)
        A1_x = torch.cat([cos_a, sin_a], dim=2)
        A2_x = torch.cat([-sin_a, cos_a], dim=2)
        transform = torch.cat([A1_x, A2_x], dim=1)
        return transform

    def forward(self, x, return_rot_matrix=False):
        gx = self.gx(F.pad(x, (1, 1, 0, 0), "replicate"))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), "replicate"))
        mag = torch.sqrt(gx * gx + gy * gy + 1e-10)
        if x.is_cuda:
            self.gk = self.gk.cuda()
        mag = mag * self.gk.unsqueeze(0).unsqueeze(0).expand_as(mag)
        ori = torch.atan2(gy, gx)
        o_big = float(self.num_ang_bins) * (ori + 1.0 * math.pi) / (2.0 * math.pi)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % self.num_ang_bins
        # bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        # wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(F.adaptive_avg_pool2d((bo0_big == i).float() * wo0_big, (1, 1)))
        ang_bins = torch.cat(ang_bins, 1).view(-1, 1, self.num_ang_bins)
        ang_bins = self.angular_smooth(ang_bins)
        values, indices = ang_bins.view(-1, self.num_ang_bins).max(1)
        angle = -((2.0 * float(np.pi) * indices.float() / float(self.num_ang_bins)) - float(math.pi))
        if return_rot_matrix:
            return self.get_rotation_matrix(angle)
        return angle

class NMS3d(nn.Module):
    def __init__(self, kernel_size = 3, threshold = 0):
        super(NMS3d, self).__init__()
        self.MP = nn.MaxPool3d(kernel_size, stride=1, return_indices=False, padding = (0, kernel_size//2, kernel_size//2))
        self.eps = 1e-5
        self.th = threshold
        return
    def forward(self, x):
        #local_maxima = self.MP(x)
        if self.th > self.eps:
            return  x * (x > self.th).float() * ((x + self.eps - self.MP(x)) > 0).float()
        else:
            return ((x - self.MP(x) + self.eps) > 0).float() * x

class NMS3dAndComposeA(nn.Module):
    def __init__(self, w=0, h=0, kernel_size=3, threshold=0, scales=None, border=3, mrSize=1.0):
        super(NMS3dAndComposeA, self).__init__()
        self.eps = 1e-7
        self.ks = 3
        self.th = threshold
        self.cube_idxs = []
        self.border = border
        self.mrSize = mrSize
        self.beta = 1.0
        self.grid_ones = torch.ones(3, 3, 3, 3)
        self.NMS3d = NMS3d(kernel_size, threshold)
        if (w > 0) and (h > 0):
            self.spatial_grid = generate_2dgrid(h, w, False).view(1, h, w, 2).permute(3, 1, 2, 0)
        else:
            self.spatial_grid = None
        return

    def forward(self, low, cur, high, num_features=0, octaveMap=None, scales=None):
        assert low.size() == cur.size() == high.size()
        # Filter responce map
        self.is_cuda = low.is_cuda
        resp3d = torch.cat([low, cur, high], dim=1)

        mrSize_border = int(self.mrSize)
        if octaveMap is not None:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.unsqueeze(1)).squeeze(1)[:, 1:2, :, :], mrSize_border) * (1.0 - octaveMap.float())
        else:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.unsqueeze(1)).squeeze(1)[:, 1:2, :, :], mrSize_border)

        num_of_nonzero_responces = (nmsed_resp > 0).float().sum()
        if num_of_nonzero_responces == 0:
            return None, None, None
        if octaveMap is not None:
            octaveMap = (octaveMap.float() + nmsed_resp.float()).byte()

        nmsed_resp = nmsed_resp.view(-1)
        if (num_features > 0) and (num_features < num_of_nonzero_responces):
            nmsed_resp, idxs = torch.topk(nmsed_resp, k=num_features)
        else:
            idxs = nmsed_resp.data.nonzero().squeeze()
            nmsed_resp = nmsed_resp[idxs]
        if len(idxs.size()) == 0:
            return None, None, None
        if len(idxs) == 0:
            return None, None, None
        # Get point coordinates grid

        if type(scales) is not list:
            self.grid = generate_3dgrid(3, self.ks, self.ks)
        else:
            self.grid = generate_3dgrid(scales, self.ks, self.ks)
        self.grid = self.grid.t().contiguous().view(3, 3, 3, 3)
        if self.spatial_grid is None:
            self.spatial_grid = generate_2dgrid(low.size(2), low.size(3), False).view(1, low.size(2), low.size(3), 2).permute(3, 1, 2, 0)
        if self.is_cuda:
            self.spatial_grid = self.spatial_grid.cuda()
            self.grid_ones = self.grid_ones.cuda()
            self.grid = self.grid.cuda()

        # residual_to_patch_center
        sc_y_x = F.conv2d(resp3d, self.grid, padding=1) / (F.conv2d(resp3d, self.grid_ones, padding=1) + 1e-8)

        ##maxima coords
        sc_y_x[0, 1:, :, :] = sc_y_x[0, 1:, :, :] + self.spatial_grid[:, :, :, 0]
        sc_y_x = sc_y_x.view(3, -1).t()
        sc_y_x = sc_y_x[idxs, :]

        min_size = float(min((cur.size(2)), cur.size(3)))
        sc_y_x[:, 0] = sc_y_x[:, 0] / min_size
        sc_y_x[:, 1] = sc_y_x[:, 1] / float(cur.size(2))
        sc_y_x[:, 2] = sc_y_x[:, 2] / float(cur.size(3))
        return nmsed_resp, sc_y_x2LAFs(sc_y_x), octaveMap

class HessianResp(nn.Module):
    def __init__(self):
        super(HessianResp, self).__init__()
        self.gx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))

        self.gy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))

        self.gxx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gxx.weight.data = torch.from_numpy(np.array([[[[1.0, -2.0, 1.0]]]], dtype=np.float32))

        self.gyy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gyy.weight.data = torch.from_numpy(np.array([[[[1.0], [-2.0], [1.0]]]], dtype=np.float32))

    def forward(self, x, scale):
        gxx = self.gxx(F.pad(x, (1, 1, 0, 0), "replicate"))
        gyy = self.gyy(F.pad(x, (0, 0, 1, 1), "replicate"))
        gxy = self.gy(F.pad(self.gx(F.pad(x, (1, 1, 0, 0), "replicate")), (0, 0, 1, 1), "replicate"))
        return torch.abs(gxx * gyy - gxy * gxy) * (scale ** 4)


# class HR_module(nn.Module):
#     def __init__(self, sigmas):
#         super(HR_module, self).__init__()
#         self.gx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
#         self.gx.weight.data = torch.from_numpy(np.array([[[[1.0, 0, -1.0]]]], dtype=np.float32))
#         self.gy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
#         self.gy.weight.data = torch.from_numpy(np.array([[[[1.0], [0], [-1.0]]]], dtype=np.float32))
#         self.gxx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
#         self.gxx.weight.data = torch.from_numpy(np.array([[[[1.0, -2.0, 1.0]]]], dtype=np.float32))
#         self.gyy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
#         self.gyy.weight.data = torch.from_numpy(np.array([[[[1.0], [-2.0], [1.0]]]], dtype=np.float32))
#
#     def forward(self, x, sigmas):
#         x = x.detach()
#         gxx = self.gxx(F.pad(x, (1, 1, 0, 0), "replicate"))
#         gyy = self.gyy(F.pad(x, (0, 0, 1, 1), "replicate"))
#         gxy = self.gy(F.pad(self.gx(F.pad(x, (1, 1, 0, 0), "replicate")), (0, 0, 1, 1), "replicate")) / 4
#         return (gxx * gyy - gxy * gxy) * (scale ** 4)

# functions take PIL image, output numpy array of the same w,h
def get_WF_from_string(weight_function):
    HR = HessianResp().cuda()

    def HRplain(pilimg, sum_to_one=False, scale=1.0):
        with torch.no_grad():
            if type(pilimg) == torch.Tensor:
                res = HR(pilimg.unsqueeze(0).mean(dim=1, keepdim=True).cuda(), scale).squeeze().cpu().data.numpy()
            else: # if pilimg is img, then RGB become (3,?,?), gray (1,?,?)
                res = HR((transforms.ToTensor()(pilimg)).unsqueeze(0).mean(dim=1, keepdim=True).cuda(), scale).squeeze().cpu().data.numpy()
            if sum_to_one:
                res /= np.sum(res)
            return res

    def HRSqrt(pilimg, sum_to_one=False, scale=1.0):
        with torch.no_grad():
            if type(pilimg) == torch.Tensor:
                res = np.sqrt(HR(pilimg.unsqueeze(0).mean(dim=1, keepdim=True).cuda(), scale).squeeze().cpu().data.numpy())
            else:
                res = np.sqrt(HR((transforms.ToTensor()(pilimg)).unsqueeze(0).mean(dim=1, keepdim=True).cuda(), scale).squeeze().cpu().data.numpy())
            if sum_to_one:
                res /= np.sum(res)
            return res

    def HRSqrt4(pilimg, sum_to_one=False, scale=1.0):
        with torch.no_grad():
            if type(pilimg) == torch.Tensor:
                res = np.sqrt(np.sqrt(HR(pilimg.unsqueeze(0).mean(dim=1, keepdim=True).cuda(), scale).squeeze().cpu().data.numpy()))
            else:
                res = np.sqrt(np.sqrt(HR((transforms.ToTensor()(pilimg)).unsqueeze(0).mean(dim=1, keepdim=True).cuda(), scale).squeeze().cpu().data.numpy()))
            if sum_to_one:
                res /= np.sum(res)
            return res

    def uniform(pilimg, sum_to_one=False):
        w1, h1 = pilimg.size
        res = np.ones((h1, w1))
        if sum_to_one:
            res /= np.sum(res)
        return res

    if weight_function == "Hessian":
        WF = HRplain
    elif weight_function == "HessianSqrt":
        WF = HRSqrt
    elif weight_function == "HessianSqrt4":
        WF = HRSqrt4
    elif weight_function in ["None", "none", 'uniform']:
        WF = uniform
    else:
        assert False, "Unknown weight function"
    return WF

def example(name='20130829_120008.png'):
    img = Image.open(name)
    img_np = np.array(img.convert("L"))
    # img_np = np.expand_dims(img_np, 0)
    # img_np = np.expand_dims(img_np, 0)
    img_np = img_np.astype(np.float) / 255
    return img_np

class SS_module(nn.Module):
    def __init__(self, sigmas):
        super(SS_module, self).__init__()
        self.features = nn.ModuleList()
        for sigma in sigmas:
            ksize = int(6*sigma)
            if ksize%2==0:
                ksize += 1

            x_cord = torch.arange(ksize).float()
            # x_grid = x_cord.repeat(ksize).view(ksize, ksize)
            # y_grid = x_grid.t()
            # xy_grid = torch.stack([x_grid, y_grid], dim=-1)

            mean = ksize // 2

            # gaussian_kernel = (1. / (np.sqrt(2. * math.pi) * sigma)) * \
            #                   torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance) )
            gker = torch.exp(-((x_cord - mean) ** 2) / (2 * (sigma ** 2)) ) / (np.sqrt(2. * math.pi) * sigma)
            gker /= torch.sum(gker)
            gker1 = gker.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
            gker2 = gker.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            self.features.append(nn.Sequential(
                nn.ReflectionPad2d((0, 0, mean, mean)),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=tuple(gker1.shape[-2:]), groups=1, bias=False),
                nn.ReflectionPad2d((mean, mean, 0, 0)),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=tuple(gker2.shape[-2:]), groups=1, bias=False)
            ))

            self.features[-1][1].weight.data = gker1.float()
            self.features[-1][3].weight.data = gker2.float()

    def forward(self, x):
        return torch.cat([f(x) for f in self.features], dim=1)

def scalespace(img_np, sigma, levels): # not used
    sigmas = [sigma**i for i in range(levels)]

    def get_blur(img_t, sigma):
        img_t = torch.from_numpy(img_np).float()
        img_t = img_t.unsqueeze(0)
        ksize = int(6*sigma)
        if ksize%2==0:
            ksize += 1

        x_cord = torch.arange(ksize).float()
        # x_grid = x_cord.repeat(ksize).view(ksize, ksize)
        # y_grid = x_grid.t()
        # xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = ksize // 2

        # gaussian_kernel = (1. / (np.sqrt(2. * math.pi) * sigma)) * \
        #                   torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance) )
        gker = torch.exp(-((x_cord - mean) ** 2) / (2 * (sigma ** 2)) ) / (np.sqrt(2. * math.pi) * sigma)
        gker /= torch.sum(gker)
        gker1 = gker.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        gker2 = gker.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        gfil1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=tuple(gker1.shape[-2:]), groups=1, bias=False)
        gfil2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=tuple(gker2.shape[-2:]), groups=1, bias=False)
        gfil1.weight.data = gker1.float()
        gfil2.weight.data = gker2.float()
        gfil1.weight.requires_grad = False
        gfil2.weight.requires_grad = False
        x = img_t
        print('A')
        x = F.pad(x, (0, 0, mean, mean), 'reflect')
        x = gfil1(x)
        x = F.pad(x, (mean, mean, 0, 0), 'reflect')
        x = gfil2(x)
        return x
    # pool = multiprocessing.Pool(4)
    # inputs = list(tqdm(pool.imap(get_blur, [img_np]*len(sigmas), [ksize]*len(sigmas), sigmas), total=levels, desc='Getting scalespace'))
    p = pathos.pools.ProcessPool(multiprocessing.cpu_count())
    outs = p.map(get_blur, img_in, sigmas)
    outs = torch.cat(outs).data.cpu().numpy()
    return outs, sigmas

def detect_positions3d(name, sigma=1.1, levels=30, nms=5, thr=0.01): # not used
    HR = HessianResp()
    img_np = example(name)
    ss, sigmas = scalespace(img_np, sigma, levels)
    ss = np.swapaxes(ss, 0, 1)
    ss = np.squeeze(ss, 0)
    ss = [HR(torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float(), b) for a, b in zip(ss, sigmas)]
    ss = torch.cat(ss).squeeze(1)

    mm = torch.nn.MaxPool3d(nms, stride=1, padding=nms // 2)(ss.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    peaks = (mm == ss) * (mm > thr)
    res = ((peaks)).data.cpu().numpy()
    z = np.nonzero(res)
    print(len(z[0]), 'detections')

    img = Image.open(name)
    draw = ImageDraw.Draw(img)
    for s, y, x in zip(*z):
        r = 3 * sigmas[s]
        draw.ellipse((x - r, y - r, x + r, y + r), outline='rgb(255,0,0)', width=1)
    img.save(os.path.splitext(name)[0]+'_sshess'+os.path.splitext(name)[1], 'JPEG')
    # img.show()

    # img_out = img_t.data.cpu().numpy()
    # img_out = (img_out * 255).astype(np.uint8)
    # img_out = np.swapaxes(np.swapaxes(np.squeeze(img_out, 0), 1, 2), 0, 2)
    # img_out = Image.fromarray(img_out)
    # img_out.show()

def one_per_window(ss, wsize, sigmas):
    z = [[], []]
    for o,c in enumerate(ss):
        for i in range(1 + c.shape[0] // wsize):
            for j in range(1 + c.shape[1] // wsize):
                a = c[i*wsize:(i+1)*wsize, j*wsize:(j+1)*wsize]
                if np.product(a.shape) == 0:
                    continue
                mm = np.argmax(a)
                if mm > 0:
                    y = i*wsize + mm // wsize
                    x = j*wsize + mm % wsize
                    z[0] += [sigmas[o]]
                    z[1] += [(x,y)]
    return np.array(z[0]), np.array(z[1])

def draw_positions(res, name, suffix=''):
    img = Image.open(name)
    draw = ImageDraw.Draw(img)
    for s, (x,y) in zip(res[0], res[1]):
        # r = 3 * s # this was in MPV course
        r = s
        draw.ellipse((x - r, y - r, x + r, y + r), outline='rgb(255,0,0)', width=1)
        # draw.rectangle((x - r, y - r, x + r, y + r), outline='rgb(255,0,0)', width=1)
    name_out = os.path.splitext(name)[0] + '_sshess_new'+suffix + os.path.splitext(name)[1]
    img.save(name_out, 'PNG')
    # img.save(os.path.splitext(name)[0] + os.path.splitext(name)[1], 'JPEG')
    print(name, '->', name_out, len(res[0]), 'detections')

def safe_dist(c, base_size):
    return 2 * c * base_size
def get_scale(c, base_size): # also radius
    return c * base_size

def gaussian_select(ss, sigma, sigmas, base_size):
    gaussis = [scipy.ndimage.gaussian_filter(c.astype(np.float), sigma) for c in ss]
    res = []
    for i in range(ss.shape[0]):
        x = np.arange(ss.shape[2])
        # print('max x:',np.max(x))
        y = np.arange(ss.shape[1])
        xv, yv = np.meshgrid(x, y)
        idxs = np.arange(0, xv.flatten()[:].shape[0])
        ch = np.random.choice(idxs, size=int(np.sum(ss[i])), p=(gaussis[i].flatten() / gaussis[i].flatten().sum()).flatten())
        res += [(i,xv.flatten()[c],yv.flatten()[c]) for c in ch]
    res = np.array(res)
    res = [np.array([get_scale(sigmas[c], base_size) for c in res[:, 0]]), res[:,[1, 2]]]
    return res

class HessDetector:
    def __init__(self, sigmas, base_size=6, fixed_MP=False):
        self.sigmas     = sigmas
        self.base_size  = base_size

        self.HR = HessianResp().cuda()
        self.SS = SS_module(self.sigmas).cuda()
        # MP = torch.nn.MaxPool2d(wsize, stride=1, padding=wsize // 2)#.cuda()
        self.MPs = [torch.nn.MaxPool2d(int(self.MP_size(c)) + 1 - (int(self.MP_size(c)) % 2), stride=1, padding=int(self.MP_size(c) / 2)).cuda() for c in self.sigmas]
        if fixed_MP:
            self.MPs = [torch.nn.MaxPool2d(int(self.MP_size_fixed(c)) + 1 - (int(self.MP_size_fixed(c)) % 2), stride=1, padding=int(self.MP_size_fixed(c) / 2)).cuda() for c in self.sigmas]

    def MP_size(self, c):
        return safe_dist(c, self.base_size)

    def MP_size_fixed(self, c):
        return safe_dist(1, self.base_size)

    def detect_positions(self, img_np: np.array, mask: np.array = None, thr=0.00016, ret_mask=False):  # wsize must be odd, thr=0.001
    ### img_np should have values in (0,1)
    ### mask is binary
        if mask is None:
            mask = np.ones(img_np.shape)
        mask = np.repeat(np.expand_dims(mask, 0), len(self.sigmas), 0)
        mask = mask.astype(np.bool)

        img_t = torch.from_numpy(img_np).float().detach().unsqueeze(0).unsqueeze(0).cuda()
        ss = self.SS(img_t).squeeze(0)
        ss = [self.HR(a.unsqueeze(0).unsqueeze(0), b) for a, b in zip(ss, self.sigmas)]
        ss = torch.cat(ss).squeeze(1)

        # mm = MP(ss.unsqueeze(0))
        mm = torch.cat([f(x.unsqueeze(0).unsqueeze(0)) for f, x in zip(self.MPs, ss)], dim=1)
        ss[(ss != mm.squeeze(0)) + (ss < thr)] = 0
        ss = ss.data.cpu().numpy()
        ss[np.logical_not(mask)] = 0

        # res = one_per_window(ss, wsize, sigmas)
        # print(ss.shape)
        if ret_mask:
            return ss > 0
        # print('counts', [np.sum(c>0) for c in ss])
        res = np.array(np.nonzero(ss))
        res = [np.array([get_scale(self.sigmas[c], self.base_size) for c in res[0, :]]), np.swapaxes(res[[2, 1], :], 0, 1)]
        return res

if __name__ == "__main__":
    # draw_positions(detect_positions(example('sunflowers.png')), 'sunflowers.png')
    # draw_positions(HessDetector(sigmas=[1.8**i for i in range(0, 6)], base_size=24, fixed_MP=True).detect_positions(example('20130930_103006.png'), mask=np.array(Image.open('00000367.png'))), '20130930_103006.png')
    draw_positions(HessDetector(sigmas=[1.8**i for i in range(5, 6)], base_size=6, fixed_MP=True).detect_positions(example('sunflowers.png')), 'sunflowers.png', '_6')
    draw_positions(HessDetector(sigmas=[1.8**i for i in range(0, 1)], base_size=24, fixed_MP=True).detect_positions(example('sunflowers.png')), 'sunflowers.png', '_24')
    # draw_positions(detect_positions(example('20130805_125425.png'), mask=np.array(Image.open('00007195.png'))), '20130805_125425.png')
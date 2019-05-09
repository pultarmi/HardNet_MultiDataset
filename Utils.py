import torch.nn.init
import numpy as np

# reshape image
np_reshape32 = lambda x: np.reshape(x, (32, 32, 1))
np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))


def zeros_like(x):
    assert x.__class__.__name__.find("Variable") != -1 or x.__class__.__name__.find("Tensor") != -1, "Object is neither a Tensor nor a Variable"
    y = torch.zeros(x.size())
    if x.is_cuda:
        y = y.cuda()
    if x.__class__.__name__ == "Variable":
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find("Tensor") != -1:
        return torch.zeros(y)


def ones_like(x):
    assert x.__class__.__name__.find("Variable") != -1 or x.__class__.__name__.find("Tensor") != -1, "Object is neither a Tensor nor a Variable"
    y = torch.ones(x.size())
    if x.is_cuda:
        y = y.cuda()
    if x.__class__.__name__ == "Variable":
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find("Tensor") != -1:
        return torch.ones(y)


def batched_forward(model, data, batch_size, **kwargs):
    n_patches = len(data)
    if n_patches > batch_size:
        bs = batch_size
        n_batches = n_patches / bs + 1
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
                first_batch_out = model(data[st:end], kwargs)
                out_size = torch.Size([n_patches] + list(first_batch_out.size()[1:]))
                # out_size[0] = n_patches
                out = torch.zeros(out_size)
                if data.is_cuda:
                    out = out.cuda()
                out[st:end] = first_batch_out
            else:
                out[st:end, :, :] = model(data[st:end], kwargs)
        return out
    else:
        return model(data, kwargs)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False




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



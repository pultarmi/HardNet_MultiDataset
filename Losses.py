import torch, sys
import torch.nn as nn
import torch.nn.functional as F

def distance_matrix_vector(anchor, positive):
    # Given batch of anchor descriptors and positive descriptors calculate distance matrix
    d1_sq = torch.norm(anchor, p=2, dim=1,keepdim = True)
    d2_sq = torch.norm(positive, p=2, dim=1,keepdim = True)
    eps = 1e-6
    return torch.sqrt(d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0))) - 2.0 * F.linear(anchor, positive) + eps)


def distance_vectors_pairwise(anchor, positive, negative=None):
    # Given batch of anchor descriptors and positive descriptors calculate distance matrix
    eps = 1e-8
    a_sq = torch.norm(anchor, p=2, dim=1,keepdim = True)
    p_sq = torch.norm(positive, p=2, dim=1,keepdim = True)
    d_a_p = torch.sqrt(a_sq + p_sq - 2 * torch.sum(anchor * positive, dim=1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(a_sq + n_sq - 2 * torch.sum(anchor * negative, dim=1) + eps)
        d_p_n = torch.sqrt(p_sq + n_sq - 2 * torch.sum(positive * negative, dim=1) + eps)
        return d_a_p, d_a_n, d_p_n
    else:
        return d_a_p


def loss_HardNetMulti(anchor, positive, 
                 margin=1.0,
                 anchor_swap=False,
                 batch_reduce="min", 
                 loss_type="triplet_margin"):
    # HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.eye(dist_matrix.size(1)).cuda()
    
    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)

    dist_without_min_on_diag = dist_matrix + eye * 10

    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask
    if batch_reduce == "min":
        min_neg = torch.min(dist_without_min_on_diag, 1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag, 0)[0]
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == "average":
        pos = pos1.repeat(anchor.size(0)).view(-1, 1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1, 1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1, 1)
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == "random":
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1, idxs.view(-1, 1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1, idxs.view(-1, 1))
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else:
        print("Unknown batch reduce mode. Try min, average or random")
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == "triplet_margin_robust":
        per_keep = 0.95
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        loss, _ = torch.sort(loss)
        loss = loss[: int(per_keep * loss.shape[0])]
    elif loss_type == "softmax":
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = -torch.log(exp_pos / exp_den)
    elif loss_type == "contrastive":
        loss = torch.clamp(margin - min_neg, min=0.0) + pos
    else:
        print("Unknown loss type. Try triplet_margin, softmax or contrastive")
        sys.exit(1)
    return loss

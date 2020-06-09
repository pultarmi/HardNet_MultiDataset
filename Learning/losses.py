import torch
import torch.nn.functional as F
from pytorch_metric_learning import miners
from pytorch_metric_learning.losses import TripletMarginLoss, CosFaceLoss
from pytorch_metric_learning.utils import loss_and_miner_utils
import numpy as np
import torch.nn as nn


def distance_matrix_vector(anchor, positive, eps = 1e-6, detach_other=False): # Given batch of anchor descriptors and positive descriptors calculate distance matrix
    d1_sq = torch.norm(anchor, p=2, dim=1, keepdim = True)
    d2_sq = torch.norm(positive, p=2, dim=1, keepdim = True)
    if detach_other:
        d2_sq = d2_sq.detach() # -> negatives do not get gradient
    # descriptors are still normalized, this is just more general formula
    return torch.sqrt(d1_sq.repeat(1, positive.size(0))**2 + torch.t(d2_sq.repeat(1, anchor.size(0)))**2 - 2.0 * F.linear(anchor, positive) + eps)

def distance_vectors_pairwise(anchor, positive, negative=None, eps = 1e-8): # Given batch of anchor descriptors and positive descriptors calculate distance matrix
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

def Huber(x, delta = 1.0):
    return delta*delta*(torch.sqrt(1.0 + (x/delta)**2) - 1)

def tripletMargin_original(anchor, positive,  # original implementation with lots of features ("too close" check, margin, detach_neg, ...)
                           margin_pos=1.0,  # if edge higher than margin_pos, change
                           anchor_swap=True,
                           batch_reduce="min",
                           loss_type="tripletMargin",
                           detach_neg=False,
                           eps = 1e-8,
                           get_edge=False,  # if -edge higher than margin_neg, change
                           block_sizes=None,
                           dup_protection=True,
                           ):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    pos = torch.diag(dist_matrix)

    if detach_neg:
        dist_matrix = distance_matrix_vector(anchor, positive, detach_other=True) + eps
    dist_without_min_on_diag = dist_matrix + torch.eye(dist_matrix.size(1)).cuda() * 10

    too_close = dist_without_min_on_diag.le(0.008).float()
    if dup_protection:
        dist_without_min_on_diag += (10 * too_close).type_as(dist_without_min_on_diag) # filter out same patches that occur in distance matrix as negatives
    if block_sizes is not None:
        dist_mask = torch.ones((anchor.shape[0], anchor.shape[0])).float().cuda() * 1000
        offset = 0
        for s in block_sizes:
            dist_mask[offset:offset+s, offset:offset+s] = 0
            offset += s
        dist_without_min_on_diag = dist_without_min_on_diag + dist_mask
    if batch_reduce == "min":
        min_neg = torch.min(dist_without_min_on_diag, dim=1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag, dim=0)[0]
            min_neg = torch.min(min_neg, min_neg2)
    elif batch_reduce == "average":
        pos = pos.repeat(anchor.size(0)).view(-1, 1).squeeze(0)
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
    else:
        assert False, "Unknown batch reduce mode. Try min, average or random"

    # too_close = min_neg.le(0.008).float()
    # min_neg += (10 * too_close).type_as(min_neg) # turns off loss for "too close" pairs, huge min_neg clamps loss to constant afterwards
    edge = pos - min_neg
    if loss_type == "tripletMargin":
        # too_bad = edge > margin_neg
        loss = torch.clamp(margin_pos + edge, min=0.0)
        # loss[too_bad] = loss[too_bad] / 4
        # loss = torch.clamp(1.0 + pos - min_neg, min=low_cut, max=high_cut)
    elif loss_type == "tripletMarginHuberInternal":
        loss = torch.clamp(margin_pos + Huber(pos) - Huber(min_neg), min=0.0)
    elif loss_type == "tripletMarginHuberExternal":
        loss = Huber(torch.clamp(margin_pos + pos - min_neg, min=0.0))
    elif loss_type == "tripletMarginHuberDouble":
        loss = Huber(torch.clamp(margin_pos + Huber(pos) - Huber(min_neg), min=0.0))
    elif loss_type == "tripletMarginRobust":
        per_keep = 0.95
        loss = torch.clamp(margin_pos + pos - min_neg, min=0.0)
        loss, _ = torch.sort(loss)
        loss = loss[: int(per_keep * loss.shape[0])]
    elif loss_type == "softmax":
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = -torch.log(exp_pos / exp_den)
    elif loss_type == "contrastive":
        loss = torch.clamp(margin_pos - min_neg, min=0.0) + pos
    else:
        assert False, "Unknown loss type. Try triplet_margin, softmax or contrastive"
    if get_edge:
        # return loss, edge
        return loss, edge, pos, min_neg
    return loss

def tripletMargin_generalized(embeddings, labels,  # this is general implementation with embeds+labels and is fast
                              margin_pos=1.0,  # if edge higher than margin_pos, change
                              # anchor_swap=True,
                              # detach_neg=False,
                              # eps = 1e-8,
                              # get_edge=False,  # if -edge higher than margin_neg, change
                              # block_sizes=None
                              ):
    # labels = torch.tensor([0, 0, 0, 0, 1, 2, 3, 3, 3, 1, 2, 2, 2, 2, 2])
    N = len(labels)
    # embeddings = torch.rand(N, 3)

    with torch.no_grad():
        dm = torch.cdist(embeddings, embeddings)

        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        # is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        HUGE_CONST = 1e5
        pos_dist_max, pos_distmax_idx = (is_pos * dm).max(dim=1)
        too_close = dm.le(0.008).float()
        # neg_dist_min, neg_distmin_idx = (HUGE_CONST * is_pos.float() + dm).min(dim=1)
        neg_dist_min, neg_distmin_idx = (is_pos*HUGE_CONST + too_close*HUGE_CONST + dm).min(dim=1)

        labels_unique = torch.unique(labels)
        NL = len(labels_unique)
        label_class_matrix = labels.unsqueeze(0).expand(NL, N).eq(labels_unique.unsqueeze(0).expand(N, NL).t())

        pos_dist_max_class = (pos_dist_max.unsqueeze(0).expand(NL, N) * label_class_matrix.float()).max(dim=1)
        neg_dist_min_class = (neg_dist_min.unsqueeze(0).expand(NL, N) + HUGE_CONST * (~label_class_matrix).float()).min(dim=1)

    # a = pos_dist_max[pos_dist_max_class[1]]
    # b = neg_dist_min[neg_dist_min_class[1]]

    a = torch.norm(embeddings[pos_dist_max_class[1]] - embeddings[pos_distmax_idx[pos_dist_max_class[1]]], dim=1)
    b = torch.norm(embeddings[neg_dist_min_class[1]] - embeddings[neg_distmin_idx[neg_dist_min_class[1]]], dim=1)

    # edge = pos_dist_max_class[0] - neg_dist_min_class[0]
    edge = a - b
    loss = torch.clamp(margin_pos + edge, min=0.0)
    return loss, edge

# def our_tripletMargin_(embeddings, labels, # I think this was my first try to write general embeds+labels loss function, but it was slow
#                        indices_tuple,
#                       margin_pos=1.0,  # if edge higher than margin_pos, change
#                       anchor_swap=True,
#                       detach_neg=False,
#                       eps = 1e-8,
#                       get_edge=False,  # if -edge higher than margin_neg, change
#                       block_sizes=None):
#     anchor_idx, positive_idx, negative_idx = indices_tuple
#     anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
#     a_p_dist = F.pairwise_distance(anchors, positives, 2)
#     a_n_dist = F.pairwise_distance(anchors, negatives, 2)
#     dist = a_p_dist - a_n_dist
#
#     loss = torch.zeros(len(set(labels)))
#     for i,l in enumerate(set(labels)):
#         # loss = loss + torch.max( dist[torch.nonzero(labels == l).squeeze(1)] )
#         edge = torch.max( dist[torch.nonzero(labels == l).squeeze(1)] )
#         loss[i] = torch.clamp(margin_pos + edge, min=0.0)
#
#     return loss, edge
#
#     dist_matrix = distance_matrix_vector(anchor, positive) + eps
#     pos = torch.diag(dist_matrix)
#
#     if detach_neg:
#         dist_matrix = distance_matrix_vector(anchor, positive, detach_other=True) + eps
#     dist_without_min_on_diag = dist_matrix + torch.eye(dist_matrix.size(1)).cuda() * 10
#
#     too_close = dist_without_min_on_diag.le(0.008).float()
#     dist_without_min_on_diag += (10 * too_close).type_as(dist_without_min_on_diag) # filter out same patches that occur in distance matrix as negatives
#     if block_sizes is not None:
#         dist_mask = torch.ones((anchor.shape[0], anchor.shape[0])).float().cuda() * 1000
#         offset = 0
#         for s in block_sizes:
#             dist_mask[offset:offset+s, offset:offset+s] = 0
#             offset += s
#         dist_without_min_on_diag = dist_without_min_on_diag + dist_mask
#     min_neg = torch.min(dist_without_min_on_diag, dim=1)[0]
#     if anchor_swap:
#         min_neg2 = torch.min(dist_without_min_on_diag, dim=0)[0]
#         min_neg = torch.min(min_neg, min_neg2)
#
#     # too_close = min_neg.le(0.008).float()
#     # min_neg += (10 * too_close).type_as(min_neg) # turns off loss for "too close" pairs, huge min_neg clamps loss to constant afterwards
#     edge = pos - min_neg
#     # too_bad = edge > margin_neg
#     loss = torch.clamp(margin_pos + edge, min=0.0)
#     # loss[too_bad] = loss[too_bad] / 4
#     # loss = torch.clamp(1.0 + pos - min_neg, min=low_cut, max=high_cut)
#     if get_edge:
#         return loss, edge
#     return loss

def get_indicator(mu, speedup=10.0, type='le'): # differentiable indicator, returns 1 if input < mu
    assert type in ['le', 'ge']
    if type in ['le']:
        return lambda x : indicator_le(x, mu, speedup)
    elif type in ['ge']:
        return lambda x : indicator_ge(x, mu, speedup)
def indicator_le(input, mu, speedup=10.0): # differentiable indicator, returns 1 if input < mu
    x = -(input - mu) # -input flips by y-axis, -mu shifts by value
    return torch.sigmoid(x * speedup)
def indicator_ge(input, mu, speedup=10.0): # differentiable indicator, returns 1 if input > mu
    x = (input - mu) # -input flips by y-axis, -mu shifts by value
    return torch.sigmoid(x * speedup)

def indicator_le(input, speedup=10.0): # differentiable indicator, returns 1 if input < mu
    return torch.sigmoid(-input * speedup) # -input flips by y-axis, -mu shifts by value
def indicator_ge(input, speedup=10.0): # differentiable indicator, returns 1 if input > mu
    return torch.sigmoid(input * speedup) # -input flips by y-axis, -mu shifts by value

# HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
def loss_AP(anchor, positive, eps = 1e-8): # we want sthing like this, with grads
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    dist_matrix = distance_matrix_vector(anchor, positive) + eps

    ss = torch.sort(dist_matrix)[0]
    dd = torch.diag(dist_matrix).view(-1,1)
    hots = (ss-dd)==0
    res = hots.nonzero()[:, 1].float()
    loss = torch.sum(torch.log(res+1))
    return loss

def loss_AP_diff(anchor, positive, speedup, eps = 1e-8): # using appwoximation of indicator
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    dist_matrix = distance_matrix_vector(anchor, positive) + eps

    dist_matrix = dist_matrix - torch.diag(dist_matrix)
    # for i in range(dist_matrix.shape[0]):
    #     f = get_indicator(dist_matrix[i,i], speedup=10.0, type='le')
    #     dist_matrix[i,:] = f(dist_matrix[i,:])
    dist_matrix = indicator_le(dist_matrix, speedup=speedup)

    loss = torch.sum(dist_matrix, dim=1)
    loss = torch.log(loss)
    return loss

class DynamicSoftMarginLoss(nn.Module): # https://github.com/lg-zhang/dynamic-soft-margin-pytorch
    def __init__(self, is_binary=False, momentum=0.01, max_dist=None, nbins=512):
        """is_binary: true if learning binary descriptor
        momentum: weight assigned to the histogram computed from the current batch
        max_dist: maximum possible distance in the feature space
        nbins: number of bins to discretize the PDF"""
        super(DynamicSoftMarginLoss, self).__init__()
        self._is_binary = is_binary

        if max_dist is None:
            # max_dist = 256 if self._is_binary else 2.0
            max_dist = 2.0

        self._momentum = momentum
        self._max_val = max_dist
        self._min_val = -max_dist
        self.register_buffer("histogram", torch.ones(nbins).cuda())

        self._stats_initialized = False
        self.current_step = None

    def _compute_distances(self, x):
        if self._is_binary:
            return self._compute_hamming_distances(x)
        else:
            return self._compute_l2_distances(x)

    def _compute_l2_distances(self, x):
        # cnt = x.size(0) // 2
        a = x[0::2,:] ### change
        p = x[1::2,:] ### change
        # a = x[:cnt, :]
        # p = x[cnt:, :]
        dmat = compute_distance_matrix_unit_l2(a, p)
        return find_hard_negatives(dmat, output_index=False, empirical_thresh=0.008)

    def _compute_hamming_distances(self, x):
        cnt = x.size(0) // 2
        ndims = x.size(1)
        a = x[:cnt, :]
        p = x[cnt:, :]

        dmat = compute_distance_matrix_hamming(
            (a > 0).float() * 2.0 - 1.0, (p > 0).float() * 2.0 - 1.0
        )
        a_idx, p_idx, n_idx = find_hard_negatives(
            dmat, output_index=True, empirical_thresh=2
        )

        # differentiable Hamming distance
        a = x[a_idx, :]
        p = x[p_idx, :]
        n = x[n_idx, :]

        pos_dist = (1.0 - a * p).sum(dim=1) / ndims
        neg_dist = (1.0 - a * n).sum(dim=1) / ndims

        # non-differentiable Hamming distance
        a_b = (a > 0).float() * 2.0 - 1.0
        p_b = (p > 0).float() * 2.0 - 1.0
        n_b = (n > 0).float() * 2.0 - 1.0

        pos_dist_b = (1.0 - a_b * p_b).sum(dim=1) / ndims
        neg_dist_b = (1.0 - a_b * n_b).sum(dim=1) / ndims

        return pos_dist, neg_dist, pos_dist_b, neg_dist_b

    def _compute_histogram(self, x, momentum):
        """update the histogram using the current batch"""
        num_bins = self.histogram.size(0)
        x_detached = x.detach()
        self.bin_width = (self._max_val - self._min_val) / (num_bins - 1)
        lo = torch.floor((x_detached - self._min_val) / self.bin_width).long()
        hi = (lo + 1).clamp(min=0, max=num_bins - 1)
        hist = x.new_zeros(num_bins)
        alpha = (
            1.0
            - (x_detached - self._min_val - lo.float() * self.bin_width)
            / self.bin_width
        )
        hist.index_add_(0, lo, alpha)
        hist.index_add_(0, hi, 1.0 - alpha)
        hist = hist / (hist.sum() + 1e-6)
        self.histogram = (1.0 - momentum) * self.histogram + momentum * hist

    def _compute_stats(self, pos_dist, neg_dist):
        hist_val = pos_dist - neg_dist
        if self._stats_initialized:
            self._compute_histogram(hist_val, self._momentum)
        else:
            self._compute_histogram(hist_val, 1.0)
            self._stats_initialized = True

    def forward(self, x):
        distances = self._compute_distances(x)
        if not self._is_binary:
            pos_dist, neg_dist = distances
            self._compute_stats(pos_dist, neg_dist)
            hist_var = pos_dist - neg_dist
        else:
            pos_dist, neg_dist, pos_dist_b, neg_dist_b = distances
            self._compute_stats(pos_dist_b, neg_dist_b)
            hist_var = pos_dist_b - neg_dist_b

        PDF = self.histogram / self.histogram.sum()
        CDF = PDF.cumsum(0)

        # lookup weight from the CDF
        bin_idx = torch.floor((hist_var - self._min_val) / self.bin_width).long()
        weight = CDF[bin_idx]

        loss = -(neg_dist * weight).mean() + (pos_dist * weight).mean()
        return loss

def compute_distance_matrix_unit_l2(a, b, eps=1e-6):
    """computes pairwise Euclidean distance and return a N x N matrix"""
    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = ((1.0 - dmat + eps) * 2.0).pow(0.5)
    return dmat

def compute_distance_matrix_hamming(a, b):
    """computes pairwise Hamming distance and return a N x N matrix"""
    dims = a.size(1)
    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = (dims - dmat) * 0.5
    return dmat

def find_hard_negatives(dmat, output_index=True, empirical_thresh=0.0):
    """a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ..."""
    cnt = dmat.size(0)

    if not output_index:
        pos = dmat.diag()

    dmat = dmat + torch.eye(cnt).to(dmat.device) * 99999  # filter diagonal
    dmat[dmat < empirical_thresh] = 99999  # filter outliers in brown dataset
    min_a, min_a_idx = torch.min(dmat, dim=0)
    min_p, min_p_idx = torch.min(dmat, dim=1)

    if not output_index:
        neg = torch.min(min_a, min_p)
        return pos, neg

    mask = min_a < min_p
    a_idx = torch.cat(
        (mask.nonzero().view(-1) + cnt, (~mask).nonzero().view(-1))
    )  # use p as anchor
    p_idx = torch.cat(
        (mask.nonzero().view(-1), (~mask).nonzero().view(-1) + cnt)
    )  # use a as anchor
    n_idx = torch.cat((min_a_idx[mask], min_p_idx[~mask] + cnt))
    return a_idx, p_idx, n_idx

def approx_hamming_distance(a, p):
    return (1.0 - a * p).sum(dim=1) * 0.5
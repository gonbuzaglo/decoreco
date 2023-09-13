import torch
import torchvision
import common_utils
from common_utils.common import flatten


def normalize_batch(x, ret_all=False):
    """ Normalize each element in batch x --> (x-mean)/std"""
    n, c, h, w = x.shape
    means = x.reshape(n * c, h * w).mean(dim=1).reshape(n, c, 1, 1)
    stds = x.reshape(n * c, h * w).std(dim=1).reshape(n, c, 1, 1)
    if ret_all:
        return x.sub(means).div(stds), means, stds
    else:
        return x.sub(means).div(stds)


def l2_dist(x, y, div_dim=False):
    """ L2 distance between x and y """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    xx = x.pow(2).sum(1).view(-1, 1)
    yy = y.pow(2).sum(1).view(1, -1)
    xy = torch.einsum('id,jd->ij', x, y)
    dists = xx + yy - 2 * xy

    if div_dim:
        N, D = x.shape
        dists /= D

    return dists


def ncc_dist(x, y, div_dim=False):
    """ Normalized Cross-Correlation distacne between x and y """
    return l2_dist(normalize_batch(x), normalize_batch(y), div_dim)


def transform_vmin_vmax_batch(x, min_max=None):
    """ Transform each image in x: [min, max] --> [0, 1]"""
    if min_max is None:
        vmin = x.data.reshape(x.shape[0], -1).min(dim=1)[0][:, None, None, None]
        vmax = x.data.reshape(x.shape[0], -1).max(dim=1)[0][:, None, None, None]
    else:
        vmin, vmax = min_max
    return (x - vmin).div(vmax - vmin)


def viz_nns(x, y, max_per_nn=None, metric='ncc', ret_all=False):
    """
    return a batch, for each image in x, its nn in y
    sorted according to closest nn
    metric: NCC
    max_per_nn: filter duplicates (leave only max_per_nn elements of y-elements)
    """

    if metric == 'ncc':
        dists = ncc_dist(x, y)
    elif metric == 'l2':
        dists = l2_dist(x, y)
    else:
        raise ValueError(f'Unknown metric={metric}')

    v, nn_idx = dists.min(dim=1)

    keep = None
    if max_per_nn is not None:
        nn_idx_vals_i = torch.stack([nn_idx, v, torch.arange(v.shape[0], device=v.device)])
        nn_idx_vals_i = [(int(a), b, int(c)) for a, b, c in nn_idx_vals_i.t().tolist()]  # bring indexes back to int
        sorted_stuff = sorted(nn_idx_vals_i)

        # filter duplicates (leave only max_per_nn from each image from y)
        counter = 0
        cur_idx = sorted_stuff[0][0]
        keep = []
        for e in sorted_stuff:
            if e[0] != cur_idx:
                cur_idx = e[0]
                counter = 0
            if counter < max_per_nn:
                keep.append(e)
                counter += 1
        # sort by best value first
        keep = sorted(keep, key=lambda q: q[1])
        # keep is now: (nn idx in y, value, idx of x)
        xx = x[torch.tensor([e[2] for e in keep])]
        yy = y[torch.tensor([e[0] for e in keep])]
        v = torch.tensor([e[1] for e in keep])
    else:
        _, sidxs = v.sort()
        xx = x[sidxs]
        yy = y[nn_idx[sidxs]]

    qq = torch.stack(flatten(list(zip(xx, yy))))
    qq = transform_vmin_vmax_batch(qq)

    if ret_all:
        return qq, v, xx, yy, keep

    return qq, v



def get_model_outputs_on_grid(model, lim=1.5, n=1000):
    x_coord = torch.linspace(-lim, lim, n)
    y_coord = torch.linspace(-lim, lim, n)
    grid = torch.stack(torch.meshgrid([x_coord, y_coord], indexing=None))
    zi = model(grid.reshape(2, -1).t().to('cuda')).reshape(n, n).cpu().data
    xi = grid[0, :, :].cpu()
    yi = grid[1, :, :].cpu()
    return xi, yi, zi

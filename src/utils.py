import pickle
import random

import numpy as np
import torch

CUDA = torch.cuda.is_available()
TRN_DEVICE = torch.device('cuda' if CUDA else 'cpu')
EVA_DEVICE = torch.device('cuda')


def load_obj(name):
    """
    Load pickle file
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


def set_seed(seed):
    """
    Set random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def spy_sparse2torch_sparse(data):
    """
    Transform scipy sparse tensor to torch sparse tensor
    """
    samples = data.shape[0]
    features = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = torch.LongTensor(np.array([coo_data.row, coo_data.col]))
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t


def sparse_dense_mul(s, d):
    """
    Matrix multiplication between sparse matrix and dense matrix
    """
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def naive_sparse2tensor(data):
    """
    Transform torch sparse tensor to torch dense tensor
    """
    return torch.FloatTensor(data.toarray())


def evaluate_accuracies_test(pred, pos_idx, ks: list, batch=False):
    """
    Evaluate accuracies for test dataset
    """
    recalls = []
    maps = []
    pred_rank = torch.topk(pred, max(ks), dim=1, sorted=True)[1]
    # pred_rank = torch.argsort(pred, dim=1, descending=True)
    pos = torch.eq(pred_rank, pos_idx.unsqueeze(1)).float()
    for k in ks:
        recall = pos[:, :k].sum()
        if not batch:
            recall = recall/pred.shape[0]
        idxs = torch.nonzero(pos[:, :k], as_tuple=True)[1]
        map = (1/(idxs + 1).float()).sum()
        if not batch:
            map = map/pred.shape[0]
        recalls.append(recall.item())
        maps.append(map.item())
    return recalls, maps


def evaluate_metrics(pred, pos_idx, bundle_item, ks: list, div: bool, score=True):
    """
    Evaluate performance in terms of recalls, maps, and frequencies
    """
    recalls, maps, freqs = [], [], []
    if score:
        pred_rank = torch.topk(pred, max(ks), dim=1, sorted=True)[1]
    else:
        pred_rank = pred
    for k in ks:
        recall, map, freq = get_metrics(pred_rank, pos_idx, k, bundle_item, div)
        recalls.append(recall)
        maps.append(map)
        freqs.append(freq)
    return recalls, maps, torch.stack(freqs)


def get_metrics(pred_rank, pos_idx, k, bundle_item, div: bool):
    """
    Get evaluation metrics
    """
    pos = torch.eq(pred_rank, pos_idx).float()
    # recall and map
    recall = pos[:, :k].sum().item()
    idxs = torch.nonzero(pos[:, :k], as_tuple=True)[1]
    map = (1 / (idxs + 1).float()).sum().item()
    # frequency
    if div:
        freq = torch.tensor(
            bundle_item[pred_rank[:, :k].flatten().cpu()].sum(axis=0)).squeeze()
    else:
        freq = torch.zeros(bundle_item.shape[1])
    return recall, map, freq


def evaluate_diversities(freqs, div: bool):
    """
    Evaluate diversities
    """
    covs, ents, ginis = [], [], []
    if div:
        for freq in freqs:
            cov = torch.count_nonzero(freq).item()
            covs.append(cov/freqs.shape[1])
            prob = freq/freq.sum()
            prob = prob.clamp(min=1e-9)
            ent = -prob*torch.log2(prob)
            ent = torch.sum(ent)
            ents.append(ent)
            gini = evaluate_gini(freq.float()).item()
            ginis.append(gini)
        return covs, ents, ginis
    else:
        covs = [0., 0., 0.]
        ents = [0., 0., 0.]
        ginis = [0., 0., 0.]
        return covs, ents, ginis


def evaluate_gini(freq, eps=1e-7):
    """
    Evaluate Gini-coefficient
    """
    freq += eps
    freq = freq.sort()[0]
    n = freq.shape[0]
    idx = torch.arange(1, n + 1)
    return (torch.sum((2 * idx - n - 1) * freq)) / (n * freq.sum())


def evaluate_freqs(pred, id_batch, bundle_item, ks: list):
    """
    Evaluate frequency
    """
    pred_rank = torch.argsort(pred, dim=1, descending=True)
    ids = torch.gather(id_batch, 1, pred_rank)
    freqs = []
    for k in ks:
        rec_bundles = torch.flatten(ids[:, :k])
        freq = np.array(bundle_item[rec_bundles.detach().cpu().numpy()].sum(0)).squeeze().astype(int)
        freqs.append(freq)
    return np.stack(freqs)


def masked_softmax(vec, mask, dim=1, epsilon=1e-7):
    """
    Compute masked softmax function
    """
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    masked_sums = torch.clamp(masked_sums, min=epsilon)
    return (masked_exps/masked_sums)


def sparse_masked_softmax(tensor, mask, dim=1, epsilon=1e-7):
    """
    Compute sparse masked softmax function
    """
    sparse_masked = sparse_dense_mul(mask, tensor)
    return torch.sparse.softmax(sparse_masked, dim=dim)


def init_weights(layer):
    """
    Initialize weights
    """
    # Xavier Initialization for weights
    size = layer.weight.size()
    fan_out = size[0]
    fan_in = size[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    layer.weight.data.normal_(0.0, std)

    # Normal Initialization for Biases
    layer.bias.data.normal_(0.0, 0.001)

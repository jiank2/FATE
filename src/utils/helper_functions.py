import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing as skpp
import torch


def encode_onehot(labels):
    """Encode label to a one-hot vector."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def row_normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv @ mx
    return mx


def row_normalize_tensor(t):
    """Row-normalize a dense torch tensor."""
    rowsum = torch.sum(t, 1)
    d_inv_sqrt = torch.pow(rowsum, -1)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(t, d_mat_inv_sqrt).transpose(0, 1))


def symmetric_normalize(mat):
    """Symmetric-normalize sparse matrix."""
    D = np.asarray(mat.sum(axis=0).flatten())
    D = np.divide(1, D, out=np.zeros_like(D), where=D != 0)
    D = sp.diags(np.asarray(D)[0, :])
    D.data = np.sqrt(D.data)
    return D @ mat @ D


def symmetric_normalize_tensor(t):
    """Symmetric-normalize a dense torch tensor."""
    rowsum = torch.sum(t, 1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(t, d_mat_inv_sqrt).transpose(0, 1), d_mat_inv_sqrt)


def normalize_feature_z_score(t):
    """Apply z-score normalization on each feature."""
    t_mean = torch.mean(t, 0)
    t_std = torch.std(t, 0)
    return (t - t_mean) / t_std


def normalize_feature_min_max(t):
    """Apply min-max normalization on each feature to the range of [0, 1]."""
    t_min = t.min(axis=0)[0]
    t_max = t.max(axis=0)[0]
    return (t - t_min).div(t_max - t_min)


def accuracy(output, labels):
    """Calculate accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_matrix_to_sparse_tensor(mat):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    values = torch.from_numpy(mat.data)
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_tensor_to_sparse_matrix(t):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    indices = t.indices()
    row, col = indices[0, :].cpu().numpy(), indices[1, :].cpu().numpy()
    values = t.values().cpu().numpy()
    mat = sp.coo_matrix((values, (row, col)), shape=(t.shape[0], t.shape[1]))
    return mat


def sparse_matrix_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def random_split(dataset):
    """Randomly split nodes into train/val/test."""
    # initialization
    mask = torch.empty(dataset.num_nodes, dtype=torch.bool).fill_(False)
    if dataset.is_ratio:
        num_train = int(dataset.ratio_train * dataset.num_nodes)
        num_val = int(dataset.ratio_val * dataset.num_nodes)
        # num_test = dataset.num_nodes - num_train - num_val
    else:
        num_train = dataset.num_train
        num_val = dataset.num_val
        # num_test = dataset.num_test

    # get indices for training
    if (not dataset.is_ratio) and dataset.split_by_class:
        train_idx = dataset.get_split_by_class(num_train_per_class=num_train)
    else:
        train_idx = torch.randperm(dataset.num_nodes)[:num_train]

    # get remaining indices
    mask[train_idx] = True
    remaining = (~mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    # get indices for validation and test
    val_idx = remaining[:num_val]
    test_idx = remaining[num_val:]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def pairwise_dist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.0:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2)
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1.0 / norm)


def jaccard_similarity(mat):
    """Get jaccard similarity matrix."""
    # make it a binary matrix
    mat_bin = mat.copy()
    mat_bin.data[:] = 1

    col_sum = mat_bin.getnnz(axis=0)
    ab = mat_bin.dot(mat_bin.T)
    aa = np.repeat(col_sum, ab.getnnz(axis=0))
    bb = col_sum[ab.indices]
    sim = ab.copy()
    sim.data /= aa + bb - ab.data
    return sim


def cosine_similarity(mat):
    """Get cosine similarity matrix."""
    mat_row_norm = skpp.normalize(mat, axis=1)
    sim = mat_row_norm.dot(mat_row_norm.T)
    return sim


def filter_similarity_matrix(sim, sigma):
    """Filter value by threshold = mean(sim) + sigma * std(sim)."""
    sim_mean = np.mean(sim.data)
    sim_std = np.std(sim.data)
    threshold = sim_mean + sigma * sim_std
    sim.data *= sim.data >= threshold  # filter values by threshold
    sim.eliminate_zeros()
    return sim


def get_similarity_matrix(mat, similarity_measure=None):
    """Get similarity matrix of nodes in specified metric."""
    if similarity_measure == "jaccard":
        return jaccard_similarity(mat.tocsc())
    elif similarity_measure == "cosine":
        return cosine_similarity(mat.tocsc())
    else:
        raise ValueError(
            "Please specify the type of similarity measure to either jaccard or cosine."
        )

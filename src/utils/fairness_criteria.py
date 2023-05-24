import math

import torch
import torch.nn as nn

from utils.helper_functions import sparse_matrix_to_sparse_tensor


class GroupFairnessKDE(nn.Module):
    """KDE estimation for group fairness"""

    def __init__(self, delta=1.0, device="cpu"):
        super(GroupFairnessKDE, self).__init__()
        self.delta = delta
        self.pi = torch.tensor(math.pi).to(device)

    def forward(
        self,
        output,
        sensitive_labels,
        bandwidth=0.1,
        tau=0.5,
    ):
        return self._get_kde_delta_sp(
            output=output,
            sensitive_labels=sensitive_labels,
            bandwidth=bandwidth,
            tau=tau,
        )

    def _get_kde_delta_sp(self, output, sensitive_labels, bandwidth=0.01, tau=0.5):
        res = 0
        p_y = self._cdf(
            output=output,
            bandwidth=bandwidth,
            tau=tau,
        )
        for sensitive_value in set(sensitive_labels.tolist()):
            p_y_s = self._cdf(
                output=output[sensitive_labels == sensitive_value],
                bandwidth=bandwidth,
                tau=tau,
            )
            res += torch.abs(p_y_s - p_y)
        return res

    def statistical_parity(self, output, sensitive_labels, bandwidth=0.1, tau=0.5):
        res = 0
        p_y = self._cdf(
            output=output,
            bandwidth=bandwidth,
            tau=tau,
        )
        print("p_y", p_y)
        for sensitive_value in set(sensitive_labels.tolist()):
            p_y_s = self._cdf(
                output=output[sensitive_labels == sensitive_value],
                bandwidth=bandwidth,
                tau=tau,
            )
            num_sensitive = sensitive_labels[sensitive_labels == sensitive_value].shape[
                0
            ]

            delta_sp_base = p_y_s - p_y
            print("delta_sp_base", delta_sp_base)
            delta_sp = (
                torch.dot(
                    self._normal_distribution(
                        (tau - output.detach()[sensitive_labels == sensitive_value])
                        / bandwidth
                    ).view(-1),
                    output[sensitive_labels == sensitive_value].view(-1),
                )
                / bandwidth
                / num_sensitive
            )
            print("delta_sp before minus", delta_sp)
            delta_sp -= (
                torch.dot(
                    self._normal_distribution((tau - output.detach()) / bandwidth).view(
                        -1
                    ),
                    output.view(-1),
                )
                / bandwidth
                / sensitive_labels.shape[0]
            )
            print("delta_sp after minus", delta_sp)

            if delta_sp_base.abs() >= self.delta:
                if delta_sp_base > 0:
                    delta_sp *= self.delta
                else:
                    delta_sp *= -self.delta
                delta_sp -= (self.delta**2) / 2
            else:
                delta_sp *= delta_sp_base / 2
            res += delta_sp
        return res

    def _normal_distribution(self, x):
        return torch.exp(-0.5 * x**2) / torch.sqrt(2 * self.pi)

    @staticmethod
    def _gaussian_q_func(x):
        a = 0.4920
        b = 0.2887
        c = 1.1893
        return torch.exp(-a * x**2 - b * x - c)

    def _cdf(self, output, bandwidth=0.1, tau=0.5):
        num_sample = len(output)
        output_tilde = (tau - output) / bandwidth
        term1 = torch.sum(self._gaussian_q_func(output_tilde[output_tilde > 0]))
        term2 = torch.sum(
            1 - self._gaussian_q_func(torch.abs(output_tilde[output_tilde < 0]))
        )
        term3 = len(output_tilde[output_tilde == 0])
        denom = term1 + term2 + 0.5 * term3
        return denom / num_sample


class INFORMForAttacker(nn.Module):
    """Bias measure for individual fairness."""

    def __init__(self, similarity_laplacian, no_cuda, device):
        super(INFORMForAttacker, self).__init__()
        try:
            self.similarity_laplacian = similarity_laplacian.tocsr()
        except:
            self.similarity_laplacian = similarity_laplacian
        self.similarity_laplacian = sparse_matrix_to_sparse_tensor(
            self.similarity_laplacian
        )
        if not no_cuda:
            self.similarity_laplacian = self.similarity_laplacian.to(device)
        self.nnz = self.similarity_laplacian.coalesce().values().shape[0]

    def forward(self, x):
        """
        x: output for nodes
        """
        x = torch.mm(
            x.permute(1, 0), torch.sparse.mm(self.similarity_laplacian, x)
        )
        bias = torch.trace(x) / self.nnz
        return bias


class INFORMForTrainer(nn.Module):
    """Bias measure for individual fairness."""

    def __init__(self, similarity_laplacian, train_idx, val_idx, test_idx, no_cuda, device):
        super(INFORMForTrainer, self).__init__()
        try:
            self.similarity_laplacian = similarity_laplacian.tocsr()
        except:
            self.similarity_laplacian = similarity_laplacian
        self.train_similarity_laplacian = sparse_matrix_to_sparse_tensor(
            self.similarity_laplacian[train_idx.cpu().numpy(), :][
                :, train_idx.cpu().numpy()
            ]
        )
        self.val_similarity_laplacian = sparse_matrix_to_sparse_tensor(
            self.similarity_laplacian[val_idx.cpu().numpy(), :][
                :, val_idx.cpu().numpy()
            ]
        )
        self.test_similarity_laplacian = sparse_matrix_to_sparse_tensor(
            self.similarity_laplacian[test_idx.cpu().numpy(), :][
                :, test_idx.cpu().numpy()
            ]
        )
        if not no_cuda:
            self.train_similarity_laplacian = self.train_similarity_laplacian.to(device)
            self.val_similarity_laplacian = self.val_similarity_laplacian.to(device)
            self.test_similarity_laplacian = self.test_similarity_laplacian.to(device)
        self.train_nnz = self.train_similarity_laplacian.coalesce().values().shape[0]
        self.val_nnz = self.val_similarity_laplacian.coalesce().values().shape[0]
        self.test_nnz = self.test_similarity_laplacian.coalesce().values().shape[0]

    def forward(self, x, stage):
        """
        x: output for nodes
        """
        if stage == "train":
            x = torch.mm(x.permute(1, 0), torch.sparse.mm(self.train_similarity_laplacian, x))
            bias = torch.trace(x) / self.train_nnz
        elif stage in ("val", "validation"):
            x = torch.mm(x.permute(1, 0), torch.sparse.mm(self.val_similarity_laplacian, x))
            bias = torch.trace(x) / self.val_nnz
        elif stage == "test":
            x = torch.mm(x.permute(1, 0), torch.sparse.mm(self.test_similarity_laplacian, x))
            bias = torch.trace(x) / self.test_nnz
        else:
            raise ValueError("stage must be one of (`train`, `val`, `validation`, `test`)!")
        return bias
    
import logging

import torch
from sklearn.metrics import f1_score, roc_auc_score


class Evaluator:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_accuracy(output, labels):
        if output.shape[1] == 1:
            preds = (output.squeeze() > 0).type_as(labels)
        else:
            preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    @staticmethod
    def _get_f1(output, labels, type="micro"):
        if output.shape[1] == 1:
            preds = (output.squeeze() > 0).type_as(labels)
        else:
            preds = output.max(1)[1].type_as(labels)
        preds = torch.flatten(preds)
        preds = preds.cpu().detach().numpy()

        labels = torch.flatten(labels)
        labels = labels.cpu().detach().numpy()
        return f1_score(labels, preds, average=type)

    @staticmethod
    def _get_roc_auc(output, labels):
        if output.shape[1] > 1:
            output = output[:, -1]
        output = torch.flatten(output)
        output = output.cpu().detach().numpy()

        labels = torch.flatten(labels)
        labels = labels.cpu().detach().numpy()
        return roc_auc_score(labels, output)

    @staticmethod
    def _get_inform_bias(output, inform_calculator, stage):
        return inform_calculator(output, stage)

    @staticmethod
    def _get_statistical_parity(output, labels, sensitive_labels):
        idx_majority = (sensitive_labels == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        idx_minority = (sensitive_labels == 0).nonzero(as_tuple=True)[0].cpu().numpy()

        if output.shape[1] == 1:
            labels_preds = (output.squeeze() > 0).type_as(labels).cpu().numpy()
        else:
            labels_preds = output.max(1)[1].type_as(labels).cpu().numpy()

        if len(idx_majority) == 0 and len(idx_minority) == 0:
            delta_sp = 0.0
        elif len(idx_minority) == 0:
            delta_sp = abs(sum(labels_preds[idx_majority]) / len(idx_majority))
        elif len(idx_majority) == 0:
            delta_sp = abs(sum(labels_preds[idx_minority]) / len(idx_minority))
        else:
            delta_sp = abs(
                sum(labels_preds[idx_minority]) / len(idx_minority)
                - sum(labels_preds[idx_majority]) / len(idx_majority)
            )
        return delta_sp

    def eval(
        self,
        loss,
        output,
        labels,
        sensitive_labels,
        idx,
        stage,
        is_individual_fairness=False,
        inform_calculator=None,
    ):
        micro_f1 = self._get_f1(output[idx], labels[idx], type="micro")
        macro_f1 = self._get_f1(output[idx], labels[idx], type="macro")
        binary_f1 = self._get_f1(output[idx], labels[idx], type="binary")
        roc_auc = self._get_roc_auc(output[idx], labels[idx])

        info = f"{stage} - loss: {loss}\tmicro_f1: {micro_f1}\tmacro_f1: {macro_f1}\tbinary_f1: {binary_f1}\troc_auc: {roc_auc}\t"

        bias = (
            (
                self._get_inform_bias(output[idx], inform_calculator, stage)
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )
            if is_individual_fairness
            else self._get_statistical_parity(
                output[idx], labels[idx], sensitive_labels[idx]
            )
        )

        info += f"bias: {bias}"

        if stage in ("validation"):
            info += "\n"

        return {
            "stage": stage,
            "loss": loss,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "binary_f1": binary_f1,
            "roc_auc": roc_auc,
            "bias": bias,
        }

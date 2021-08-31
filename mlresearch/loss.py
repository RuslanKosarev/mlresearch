
import torch
from torch.nn import functional as F


def bce_loss(y_real, y_pred):
    """
    loss = torch.nn.BCEWithLogitsLoss()
    loss(y_real, y_pred)

    :param y_real:
    :param y_pred:
    :return:
    """
    loss = torch.maximum(y_real, torch.tensor(0)) - y_real * y_pred + torch.log(torch.tensor(1) + torch.exp(-torch.abs(y_real)))

    return loss.mean()


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction: str = 'mean'):
        super().__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise NotImplementedError(f'Reduction {reduction} not implemented.')

        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        p_t = torch.where(target == 1, x, 1-x)
        loss = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        loss = torch.where(target == 1, loss * self.alpha, loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


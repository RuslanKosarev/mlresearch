
import torch


def bce_loss(y_real, y_pred):
    f = torch.maximum(y_real, torch.tensor(0)) - y_real * y_pred + torch.log(torch.tensor(1) + torch.exp(-torch.abs(y_real)))
    return f.mean()


import torch


def binary_cross_entropy_with_logits(input, target): # noqa
    """
    loss = torch.nn.BCEWithLogitsLoss()
    loss(y_real, target)

    loss = -target*torch.log(torch.sigmoid(input)) - (1 - target)*torch.log(1 - torch.sigmoid(input))

    :param input:
    :param target:
    :return:
    """
    loss = torch.clamp(input, 0) - input * target + torch.log(1 + torch.exp(-torch.abs(input)))

    return loss.mean()


def focal_loss(y_real, y_pred, eps=1e-8, gamma=2):
    # loss = torch.maximum(y_real, torch.tensor(0)) - y_real * y_pred + torch.log(torch.tensor(1) + torch.exp(-torch.abs(y_real)))

    return loss.mean()


# from torch.autograd import Variable
# class FocalLoss(torch.nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         else:
#             self.alpha = torch.Tensor([alpha, 1-alpha])
#
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
#
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()

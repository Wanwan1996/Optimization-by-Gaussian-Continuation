import torch
from torch.autograd.variable import Variable
from torch.nn.modules.module import Module
import numpy as np


class NSSLoss(Module):
    def __init__(self, size_average=True):
        super(NSSLoss, self).__init__()
        self.reduce = size_average

    def forward(self, y_pred, y_true):
        assert not y_true.requires_grad

        y_pred = y_pred.contiguous().view((y_pred.size()[0], -1))
        y_true = y_true.contiguous().view((y_true.size()[0], -1))

        y_mean = torch.mean(y_pred, dim=(-1,), keepdim=True)
        y_std = torch.std(y_pred, dim=-1, keepdim=True)

        y_pred_norm = torch.div((y_pred - y_mean), torch.add(y_std, 1e-7))

        loss = -torch.div(torch.sum(torch.mul(y_pred_norm, y_true), dim=(-1,), keepdim=True),
                          torch.sum(y_true, dim=(-1,), keepdim=True))
        if self.reduce:
            return loss.mean()
        else:
            return loss.sum()


class CCLoss(Module):
    def __init__(self, size_average=True):
        super(CCLoss, self).__init__()
        self.reduce = size_average

    def forward(self, y_pred, y_true):
        assert not y_true.requires_grad

        N = torch.mul(y_pred.size()[2], y_pred.size()[3])
        sum_prod = torch.sum(torch.mul(y_true, y_pred), dim=(2, 3))

        sum_x = torch.sum(y_true, dim=(2, 3))
        sum_y = torch.sum(y_pred, dim=(2, 3))
        sum_x_square = torch.sum(y_true.pow(2), dim=(2, 3))
        sum_y_square = torch.sum(y_pred.pow(2), dim=(2, 3))

        num = sum_prod - torch.div(torch.mul(sum_x, sum_y), N)
        den = torch.sqrt(torch.mul((sum_x_square - sum_x.pow(2) / N),
                                   (sum_y_square - sum_y.pow(2) / N)))

        loss = -torch.div(num, den)

        if self.reduce:
            return loss.mean()
        else:
            return loss.sum()


class KLDLoss(Module):
    def __init__(self, size_average=True):
        super(KLDLoss, self).__init__()
        self.reduce = size_average

    def forward(self, y_pred, y_true):
        assert not y_true.requires_grad

        sum_y_true = torch.sum(y_true, dim=(2, 3), keepdim=True)
        sum_y_pred = torch.sum(y_pred, dim=(2, 3), keepdim=True)

        y_pred = torch.div(y_pred, torch.add(sum_y_pred, 1e-7))
        y_true = torch.div(y_true, torch.add(sum_y_true, 1e-7))

        loss = torch.sum(
            torch.mul(y_true,
                      torch.log(torch.add(torch.div(y_true, torch.add(y_pred, 1e-7)),
                                          1e-7))), dim=(2, 3))

        if self.reduce:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':

    a = Variable(torch.FloatTensor(np.arange(24).reshape((2, 1, 3, 4))))
    print(a.requires_grad)
    print('a = {}\n'.format(a))
    print(a.size())

    a_max = torch.max((torch.max(a, dim=2)[0]), dim=2)[0]
    print('a_max size is {}'.format(a_max.size()))
    print('a_max = {}\n'.format(a_max))

    recon_a = torch.unsqueeze(torch.unsqueeze(a_max, dim=-1).expand(-1, -1, 3), dim=-1).expand(-1, -1, -1, 4)
    print('recon_a size is {}'.format(recon_a.size()))
    print('recon_a = {}\n'.format(recon_a))

    a_nom = a / recon_a
    print('a_nom size is {}'.format(a_nom.size()))
    print('a_nom = {}\n'.format(a_nom))

    a_flatten = a_nom.view(a.size()[0], -1)
    print('a_flatten size is {}'.format(a_flatten.size()))
    print('a_flatten = {}\n'.format(a_flatten))

    a_nom_mean = torch.mean(a_flatten, dim=(-1,), keepdim=True)
    print('mean size is {}'.format(a_nom_mean.size()))
    print('mean = {}'.format(a_nom_mean))
    a_nom_std = torch.std(a_flatten, dim=-1, keepdim=True)
    print('mean size is {}'.format(a_nom_std.size()))
    print('std = {}\n'.format(a_nom_std))

    recon_a_mean = torch.unsqueeze(torch.unsqueeze(a_nom_mean, dim=-1).expand(-1, -1, 3), dim=-1).expand(-1, -1, -1, 4)
    recon_a_std = torch.unsqueeze(torch.unsqueeze(a_nom_std, dim=-1).expand(-1, -1, 3), dim=-1).expand(-1, -1, -1, 4)
    print('reconstructed mean is {}'.format(recon_a_mean))
    print('reconstructed std is {}\n'.format(recon_a_std))

    norm_a = (a_nom - recon_a_mean) / (recon_a_std + 1e-7)
    print('norm size is {}'.format(norm_a.size()))
    print('norm = {}\n'.format(norm_a))

    b = Variable(torch.FloatTensor(2, 1, 3, 4).uniform_(0, 1), requires_grad=False)
    print('b = {}'.format(b))
    norm_b = torch.bernoulli(b)
    print('ground truth of b = {}\n'.format(norm_b))

    nss = (torch.sum((norm_a * norm_b), dim=(2, 3)) / torch.sum(norm_b, dim=(2, 3)))
    print('nss size is {}\n'.format(nss.size()))
    print('nss = {}\n'.format(nss))
    print('nss_mean = {}\n'.format(nss.mean()))

    loss_ = NSSLoss(size_average=True)(a, norm_b)
    print('loss_ = {}\n'.format(loss_))
    print('loss_ mean = {}'.format(loss_.mean()))

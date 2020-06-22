import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict
from salicon_config import *


__all__ = ['DenseNet', 'DilatedDenseNet', 'MutiscaleDenseNet', 'MutiscaleDilatedDenseNet',
           'densenet121', 'dilated_densenet121', 'densenet121_pred', 'dilated_densenet121_pred',
           'densenet169', 'densenet169_pool', 'dilated_densenet169', 'dilated_densenet169_pool',
           'ml_densenet169', 'ml_densenet169_pool', 'ml_dilated_densenet169', 'ml_dilated_densenet169_pool',
           'densenet169_pred', 'densenet169_pool_pred', 'dilated_densenet169_pred', 'dilated_densenet169_pool_pred',
           'ml_densenet169_pred', 'ml_densenet169_pool_pred', 'ml_dilated_densenet169_pred',
           'ml_dilated_densenet169_pool_pred']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat((x, new_features), 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _DilatedDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilated_rate):
        super(_DilatedDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=dilated_rate, dilation=dilated_rate, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DilatedDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat((x, new_features), 1)


class _DilatedDenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilated_rate):
        super(_DilatedDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DilatedDenseLayer(num_input_features + i * growth_rate, growth_rate,
                                       bn_size, drop_rate, dilated_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _TransitionPool(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionPool, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('pad', nn.ZeroPad2d((1, 0, 0, 1)))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=1))


class _TransitionWithoutPool(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionWithoutPool, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class DenseNet(nn.Sequential):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            else:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _TransitionWithoutPool(num_input_features=num_features,
                                                   num_output_features=num_features // 2)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))


class DenseNetPool(nn.Sequential):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNetPool, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            else:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _TransitionPool(num_input_features=num_features,
                                            num_output_features=num_features // 2)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))


class DilatedDenseNet(nn.Sequential):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DilatedDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 1:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _TransitionWithoutPool(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 2:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=2)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _TransitionWithoutPool(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 3:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=4)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))


class DilatedDenseNetPool(nn.Sequential):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DilatedDenseNetPool, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            if i == 1:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 2:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=2)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 3:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=4)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))


class MutiscaleDenseNet(nn.Module):
    r"""ML_Densenet model, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(MutiscaleDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        num_block2_features = num_init_features
        num_block3_features = num_init_features
        # num_block4_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 1:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                num_block2_features = num_features
                trans = _TransitionWithoutPool(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 2:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                num_block3_features = num_features
                trans = _TransitionWithoutPool(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 3:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                # num_block4_features = num_features
            else:
                raise ValueError("Wrong block config!")

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # feature integration
        self.integ2 = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_block2_features)),
            ('relu0', nn.ReLU(inplace=True))]))
        #    ('conv0', nn.Conv2d(num_block2_features, 512, kernel_size=1, stride=1, bias=False)),
        #    ('norm1', nn.BatchNorm2d(512)),
        #    ('relu1', nn.ReLU(inplace=True))]))

        self.integ3 = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_block3_features)),
            ('relu0', nn.ReLU(inplace=True))]))
        #    ('conv0', nn.Conv2d(num_block3_features, 512, kernel_size=1, stride=1, bias=False)),
        #    ('norm1', nn.BatchNorm2d(512)),
        #    ('relu1', nn.ReLU(inplace=True))]))

        # self.integ4 = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(num_block4_features, 512, kernel_size=1, stride=1, bias=False)),
        #     ('norm0', nn.BatchNorm2d(512)),
        #     ('relu0', nn.ReLU(inplace=True))]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv0 = self.features.conv0(x)
        norm0 = self.features.norm0(conv0)
        relu0 = self.features.relu0(norm0)
        pool0 = self.features.pool0(relu0)

        denseblock1 = self.features.denseblock1(pool0)
        transition1 = self.features.transition1(denseblock1)

        denseblock2 = self.features.denseblock2(transition1)
        transition2 = self.features.transition2(denseblock2)

        denseblock3 = self.features.denseblock3(transition2)
        transition3 = self.features.transition3(denseblock3)

        denseblock4 = self.features.denseblock4(transition3)
        denseblock4 = self.features.norm5(denseblock4)
        denseblock4 = self.features.relu5(denseblock4)

        integ2 = self.integ2(denseblock2)
        integ3 = self.integ3(denseblock3)
        # integ4 = self.integ4(denseblock4)

        out = torch.cat((integ2, integ3, denseblock4), 1)

        return out


class MutiscaleDenseNetPool(nn.Module):
    r"""ML_Densenet model, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(MutiscaleDenseNetPool, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        num_block2_features = num_init_features
        num_block3_features = num_init_features
        # num_block4_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 1:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                num_block2_features = num_features
                trans = _TransitionPool(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 2:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                num_block3_features = num_features
                trans = _TransitionPool(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 3:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                # num_block4_features = num_features
            else:
                raise ValueError("Wrong block config!")

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # feature integration
        self.integ2 = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_block2_features)),
            ('relu0', nn.ReLU(inplace=True))]))
        #    ('conv0', nn.Conv2d(num_block2_features, 512, kernel_size=1, stride=1, bias=False)),
        #    ('norm1', nn.BatchNorm2d(512)),
        #    ('relu1', nn.ReLU(inplace=True))]))

        self.integ3 = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_block3_features)),
            ('relu0', nn.ReLU(inplace=True))]))
        #    ('conv0', nn.Conv2d(num_block3_features, 512, kernel_size=1, stride=1, bias=False)),
        #    ('norm1', nn.BatchNorm2d(512)),
        #    ('relu1', nn.ReLU(inplace=True))]))

        # self.integ4 = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(num_block4_features, 512, kernel_size=1, stride=1, bias=False)),
        #     ('norm0', nn.BatchNorm2d(512)),
        #     ('relu0', nn.ReLU(inplace=True))]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv0 = self.features.conv0(x)
        norm0 = self.features.norm0(conv0)
        relu0 = self.features.relu0(norm0)
        pool0 = self.features.pool0(relu0)

        denseblock1 = self.features.denseblock1(pool0)
        transition1 = self.features.transition1(denseblock1)

        denseblock2 = self.features.denseblock2(transition1)
        transition2 = self.features.transition2(denseblock2)

        denseblock3 = self.features.denseblock3(transition2)
        transition3 = self.features.transition3(denseblock3)

        denseblock4 = self.features.denseblock4(transition3)
        denseblock4 = self.features.norm5(denseblock4)
        denseblock4 = self.features.relu5(denseblock4)

        integ2 = self.integ2(denseblock2)
        integ3 = self.integ3(denseblock3)
        # integ4 = self.integ4(denseblock4)

        out = torch.cat((integ2, integ3, denseblock4), 1)

        return out


class MutiscaleDilatedDenseNet(nn.Sequential):
    r"""ML_Dilated_Densenet model, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
        """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(MutiscaleDilatedDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        num_block2_features = num_init_features
        num_block3_features = num_init_features
        # num_block4_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 1:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                num_block2_features = num_features
                trans = _TransitionWithoutPool(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 2:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=2)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                num_block3_features = num_features
                trans = _TransitionWithoutPool(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 3:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=4)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                # num_block4_features = num_features
            else:
                raise ValueError("Wrong block config!")

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # feature integration
        self.integ2 = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_block2_features)),
            ('relu0', nn.ReLU(inplace=True))]))
        #    ('conv0', nn.Conv2d(num_block2_features, 1280, kernel_size=1, stride=1, bias=False)),
        #    ('norm1', nn.BatchNorm2d(1280)),
        #    ('relu1', nn.ReLU(inplace=True))]))

        self.integ3 = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_block3_features)),
            ('relu0', nn.ReLU(inplace=True))]))
        #    ('conv0', nn.Conv2d(num_block3_features, 1280, kernel_size=1, stride=1, bias=False)),
        #    ('norm1', nn.BatchNorm2d(1280)),
        #    ('relu1', nn.ReLU(inplace=True))]))

        # self.integ4 = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(num_block4_features, 512, kernel_size=1, stride=1, bias=False)),
        #     ('norm1', nn.BatchNorm2d(512)),
        #     ('relu1', nn.ReLU(inplace=True))]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv0 = self.features.conv0(x)
        norm0 = self.features.norm0(conv0)
        relu0 = self.features.relu0(norm0)
        pool0 = self.features.pool0(relu0)

        denseblock1 = self.features.denseblock1(pool0)
        transition1 = self.features.transition1(denseblock1)

        denseblock2 = self.features.denseblock2(transition1)
        transition2 = self.features.transition2(denseblock2)

        denseblock3 = self.features.denseblock3(transition2)
        transition3 = self.features.transition3(denseblock3)

        denseblock4 = self.features.denseblock4(transition3)
        denseblock4 = self.features.norm5(denseblock4)
        denseblock4 = self.features.relu5(denseblock4)

        integ2 = self.integ2(denseblock2)
        integ3 = self.integ3(denseblock3)
        # integ4 = self.integ4(denseblock4)

        out = torch.cat((integ2, integ3, denseblock4), 1)

        return out


# class MutiscaleDilatedDenseNet(nn.Sequential):
#     r"""ML_Dilated_Densenet model, based on
#         `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
#
#         Args:
#             growth_rate (int) - how many filters to add each layer (`k` in paper)
#             block_config (list of 4 ints) - how many layers in each pooling block
#             num_init_features (int) - the number of filters to learn in the first convolution layer
#             bn_size (int) - multiplicative factor for number of bottle neck layers
#               (i.e. bn_size * k features in the bottleneck layer)
#             drop_rate (float) - dropout rate after each dense layer
#         """
#
#     def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
#                  num_init_features=64, bn_size=4, drop_rate=0):
#
#         super(MutiscaleDilatedDenseNet, self).__init__()
#
#         # First convolution
#         self.features = nn.Sequential(OrderedDict([
#             ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
#             ('norm0', nn.BatchNorm2d(num_init_features)),
#             ('relu0', nn.ReLU(inplace=True)),
#             ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#         ]))
#
#         # Each denseblock
#         num_features = num_init_features
#         # num_block2_features = num_init_features
#         # num_block3_features = num_init_features
#         num_block4_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             if i == 0:
#                 block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                     bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
#                 self.features.add_module('denseblock%d' % (i + 1), block)
#                 num_features = num_features + num_layers * growth_rate
#                 trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = num_features // 2
#             elif i == 1:
#                 block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                     bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
#                 self.features.add_module('denseblock%d' % (i + 1), block)
#                 num_features = num_features + num_layers * growth_rate
#                 trans = _TransitionWithoutPool(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = num_features // 2
#             elif i == 2:
#                 block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
#                                            growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=2)
#                 self.features.add_module('denseblock%d' % (i + 1), block)
#                 num_features = num_features + num_layers * growth_rate
#                 trans = _TransitionWithoutPool(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = num_features // 2
#             elif i == 3:
#                 block_branch1 = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
#                                                    dilated_rate=4)
#                 self.features.add_module('denseblock%d' % (i + 1), block_branch1)
#
#                 block_branch2 = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
#                                                    dilated_rate=8)
#                 self.features.add_module('denseblock%d_branch2' % (i + 1), block_branch2)
#
#                 block_branch3 = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
#                                                    dilated_rate=16)
#                 self.features.add_module('denseblock%d_branch3' % (i + 1), block_branch3)
#
#                 num_features = num_features + num_layers * growth_rate
#                 num_block4_features = num_features
#             else:
#                 raise ValueError("Wrong block config!")
#
#         # Final batch norm
#         self.features.add_module('norm5', nn.BatchNorm2d(num_features))
#         self.features.add_module('relu5', nn.ReLU(inplace=True))
#
#         # feature integration
#         self.integ = nn.Sequential(OrderedDict([
#             ('conv_integ', nn.Conv2d(num_block4_features * 3, num_block4_features,
#                                      kernel_size=1, stride=1, bias=False)),
#             ('norm_integ', nn.BatchNorm2d(num_block4_features)),
#             ('relu_integ', nn.ReLU(inplace=True))]))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         conv0 = self.features.conv0(x)
#         norm0 = self.features.norm0(conv0)
#         relu0 = self.features.relu0(norm0)
#         pool0 = self.features.pool0(relu0)
#
#         denseblock1 = self.features.denseblock1(pool0)
#         transition1 = self.features.transition1(denseblock1)
#
#         denseblock2 = self.features.denseblock2(transition1)
#         transition2 = self.features.transition2(denseblock2)
#
#         denseblock3 = self.features.denseblock3(transition2)
#         transition3 = self.features.transition3(denseblock3)
#
#         denseblock4 = self.features.denseblock4(transition3)
#         denseblock4_branch2 = self.features.denseblock4_branch2(transition3)
#         denseblock4_branch3 = self.features.denseblock4_branch3(transition3)
#
#         denseblock4 = self.features.norm5(denseblock4)
#         denseblock4_branch2 = self.features.norm5(denseblock4_branch2)
#         denseblock4_branch3 = self.features.norm5(denseblock4_branch3)
#
#         denseblock4 = self.features.relu5(denseblock4)
#         denseblock4_branch2 = self.features.relu5(denseblock4_branch2)
#         denseblock4_branch3 = self.features.relu5(denseblock4_branch3)
#
#         denseblock4 = torch.cat((denseblock4, denseblock4_branch2, denseblock4_branch3), 1)
#
#         out = self.integ(denseblock4)
#
#         return out


class MutiscaleDilatedDenseNetPool(nn.Sequential):
    r"""ML_Dilated_Densenet model, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
        """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(MutiscaleDilatedDenseNetPool, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        num_block2_features = num_init_features
        num_block3_features = num_init_features
        # num_block4_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 1:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                num_block2_features = num_features
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 2:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=2)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                num_block3_features = num_features
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == 3:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=4)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                # num_block4_features = num_features
            else:
                raise ValueError("Wrong block config!")

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('upsampling5', Interpolate(size=(60, 80), align_corners=True))

        # feature integration
        self.integ2 = nn.Sequential(OrderedDict([
             ('norm0', nn.BatchNorm2d(num_block2_features)),
             ('relu0', nn.ReLU(inplace=True))]))
        #     ('conv0', nn.Conv2d(num_block2_features, 1280, kernel_size=1, stride=1, bias=False)),
        #     ('norm1', nn.BatchNorm2d(1280)),
        #     ('relu1', nn.ReLU(inplace=True))]))

        self.integ3 = nn.Sequential(OrderedDict([
             ('norm0', nn.BatchNorm2d(num_block3_features)),
             ('relu0', nn.ReLU(inplace=True)),
             ('upsampling', Interpolate(size=(60, 80), align_corners=True))]))
        #     ('conv0', nn.Conv2d(num_block3_features, 1280, kernel_size=1, stride=1, bias=False)),
        #     ('norm1', nn.BatchNorm2d(1280)),
        #     ('relu1', nn.ReLU(inplace=True))]))

        # self.integ4 = nn.Sequential(OrderedDict([
        #      ('conv0', nn.Conv2d(num_block4_features, 512, kernel_size=1, stride=1, bias=False)),
        #      ('norm1', nn.BatchNorm2d(512)),
        #      ('relu1', nn.ReLU(inplace=True))]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv0 = self.features.conv0(x)
        norm0 = self.features.norm0(conv0)
        relu0 = self.features.relu0(norm0)
        pool0 = self.features.pool0(relu0)

        denseblock1 = self.features.denseblock1(pool0)
        transition1 = self.features.transition1(denseblock1)

        denseblock2 = self.features.denseblock2(transition1)
        transition2 = self.features.transition2(denseblock2)

        denseblock3 = self.features.denseblock3(transition2)
        transition3 = self.features.transition3(denseblock3)

        denseblock4 = self.features.denseblock4(transition3)
        denseblock4 = self.features.norm5(denseblock4)
        denseblock4 = self.features.relu5(denseblock4)
        denseblock4 = self.features.upsampling5(denseblock4)

        integ2 = self.integ2(denseblock2)
        integ3 = self.integ3(denseblock3)
        # integ4 = self.integ4(denseblock4)

        out = torch.cat((integ2, integ3, denseblock4), 1)

        return out


class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x


class Prediction(nn.Sequential):

    def __init__(self, num_input_features, num_features, shape_out):
        super(Prediction, self).__init__()
        self.add_module('conv1_pred', nn.Conv2d(num_input_features, num_features, kernel_size=3, padding=1))
        self.add_module('relu1_pred', nn.ReLU(inplace=True))
        self.add_module('conv2_pred', nn.Conv2d(num_features, 1, kernel_size=1))
        self.add_module('relu2_pred', nn.ReLU(inplace=True))
        self.add_module('pred_layer', Interpolate(size=shape_out, align_corners=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class DenseImageSalPre(nn.Module):

    def __init__(self, encode_model, pretrained, num_in_features, num_out_features, shape_out):
        super(DenseImageSalPre, self).__init__()
        self.encoder = encode_model(pretrained)
        self.predictor = Prediction(num_in_features, num_out_features, shape_out)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def dilated_densenet121(pretrained=False, **kwargs):
    r"""Dilated_Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                            drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def densenet169_pool(pretrained=False, **kwargs):
    r"""Densenet-169 model with pooling layer from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNetPool(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                         drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def dilated_densenet169(pretrained=False, **kwargs):
    r"""Dilated Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                            drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def dilated_densenet169_pool(pretrained=False, **kwargs):
    r"""Dilated Densenet-169 model with pooling layer from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedDenseNetPool(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                                drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def ml_densenet169(pretrained=False, **kwargs):
    r"""ML_Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MutiscaleDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                              drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def ml_densenet169_pool(pretrained=False, **kwargs):
    r"""ML_Densenet-169 model with pooling layer from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MutiscaleDenseNetPool(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                                  drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def ml_dilated_densenet169(pretrained=False, **kwargs):
    r"""ML_Dilated_Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MutiscaleDilatedDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                                     drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


# def ml_dilated_densenet169(pretrained=False, **kwargs):
#     r"""ML_Dilated_Densenet-169 model from
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = MutiscaleDilatedDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
#                                      drop_rate=drop_rate_, **kwargs)
#     if pretrained:
#         pattern = re.compile(
#             r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
#         state_dict = torch.load(pretrained_model)
#         # model with old keys
#         # print(state_dict.keys())
#         for key in list(state_dict.keys()):
#             res = pattern.match(key)
#             if res:
#                 new_key = res.group(1) + res.group(2)
#                 state_dict[new_key] = state_dict[key]
#                 del state_dict[key]
#         model.load_state_dict(state_dict, strict=False)
#         # model with new keys
#         # print(state_dict.keys())
#
#         # whole model parameters
#         model_para = model.state_dict().copy()
#         # print(model_para.keys())
#
#         # update parameters of dense block branch 2 and 3 with same as branch 1
#         for key, value in model.features.denseblock4.state_dict().items():
#             # print(key)
#             # print(value)
#             model_para['features.denseblock4_branch2.{}'.format(key)] = value
#             model_para['features.denseblock4_branch3.{}'.format(key)] = value
#             # print(model_para['features.denseblock4_branch2.{}'.format(key)])
#         # print(denseblock4_branch2_para['denselayer1.conv1.weight'])
#         # model_para = model.state_dict().copy
#         model_dict = model.state_dict()
#         model_dict.update(model_para)
#         model.load_state_dict(model_dict, strict=False)
#         # print(model.features.denseblock4_branch2.denselayer1.conv1.weight)
#     return model


def ml_dilated_densenet169_pool(pretrained=False, **kwargs):
    r"""ML_Dilated_Densenet-169 model with pooling layer from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MutiscaleDilatedDenseNetPool(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                                         drop_rate=drop_rate_, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_model)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def densenet121_pred(pretrained=False):
    """Constructs a DenseNet-121 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(densenet121, pretrained, 1024, 512, shape_out_)
    
    return model


def dilated_densenet121_pred(pretrained=False):
    """Constructs a Dilated DenseNet-121 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(dilated_densenet121, pretrained, 1024, 512, shape_out_)

    return model


def densenet169_pred(pretrained=False):
    """Constructs a DenseNet-169 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(densenet169, pretrained, 1664, 512, shape_out_)

    return model


def densenet169_pool_pred(pretrained=False):
    """Constructs a DenseNet-169 model with pooling layer for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(densenet169_pool, pretrained, 1664, 512, shape_out_)

    return model


def dilated_densenet169_pred(pretrained=False):
    """Constructs a Dilated DenseNet-169 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(dilated_densenet169, pretrained, 1664, 512, shape_out_)

    return model


def dilated_densenet169_pool_pred(pretrained=False):
    """Constructs a Dilated DenseNet-169 model with pooling layer for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(dilated_densenet169_pool, pretrained, 1664, 512, shape_out_)

    return model


def ml_densenet169_pred(pretrained=False):
    """Constructs a ML_DenseNet-169 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(ml_densenet169, pretrained, 3456, 512, shape_out_)

    return model


def ml_densenet169_pool_pred(pretrained=False):
    """Constructs a ML_DenseNet-169 model with pooling layer for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(ml_densenet169_pool, pretrained, 3456, 512, shape_out_)

    return model


def ml_dilated_densenet169_pred(pretrained=False):
    """Constructs a ML_Dilated_DenseNet-169 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(ml_dilated_densenet169, pretrained, 3456, 512, shape_out_)

    return model


def ml_dilated_densenet169_pool_pred(pretrained=False):
    """Constructs a ML_Dilated_DenseNet-169 model with pooling layer for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseImageSalPre(ml_dilated_densenet169_pool, pretrained, 3456, 512, shape_out_)

    return model


if __name__ == '__main__':
    # Check weights load and feature size:
    # base network densenet-169
    # model_dense169 = densenet169(pretrained=True)

    # model_dense169_pred = densenet169_pred(pretrained=True)
    # summary(model_dense169_pred, (3, 480, 640))
    # model_dilated_dense169_pred = dilated_densenet169_pred(pretrained=True)
    # summary(model_dilated_dense169_pred, (3, 480, 640))
    # model_dilated_dense169_pool_pred = dilated_densenet169_pool_pred(pretrained=True)
    # summary(model_dilated_dense169_pool_pred, (3, 480, 640))

    # model_ml_dense169_pred = ml_densenet169_pred(pretrained=True)
    # summary(model_ml_dense169_pred, (3, 480, 640))
    model_ml_dilated_dense169_pred = ml_dilated_densenet169_pred(pretrained=True)
    # model_ml_dilated_dense169_pred_1 = ml_dilated_densenet169_pred(pretrained=True)
    summary(model_ml_dilated_dense169_pred, (3, 480, 640))
    # model_ml_dilated_dense169_pool_pred = ml_dilated_densenet169_pool_pred(pretrained=True)
    # summary(model_ml_dilated_dense169_pool_pred, (3, 480, 640))

    # base parameters
    # para_dense169 = model_dense169.state_dict()

    # para_dense169_pred = model_dense169_pred.state_dict()
    # para_dilated_dense169_pred = model_dilated_dense169_pred.state_dict()
    # para_dilated_dense169_pool_pred = model_dilated_dense169_pool_pred.state_dict()

    # para_ml_dense169_pred = model_ml_dense169_pred.state_dict()
    # para_ml_dilated_dense4_b1_pred = model_ml_dilated_dense169_pred.encoder.features.denseblock4.state_dict()
    # para_ml_dilated_dense4_b2_pred = model_ml_dilated_dense169_pred.encoder.features.denseblock4_branch2.state_dict()
    # para_ml_dilated_dense4_b3_pred = model_ml_dilated_dense169_pred.encoder.features.denseblock4_branch2.state_dict()

    # para_ml_dilated_dense169_pred_1 = model_ml_dilated_dense169_pred_1.state_dict()
    # para_ml_dilated_dense169_pool_pred = model_ml_dilated_dense169_pool_pred.state_dict()

    # for key_, v_ in para_ml_dilated_dense4_b1_pred.items():
    #     print(key_)
    #     if torch.equal(para_ml_dilated_dense4_b2_pred[key_],
    #                    para_ml_dilated_dense4_b1_pred[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    #     if torch.equal(para_ml_dilated_dense4_b3_pred[key_],
    #                    para_ml_dilated_dense4_b1_pred[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))

    # for key_, v_ in para_ml_dilated_dense169_pred.items():
    #     print(key_)
    # print(para_ml_dilated_dense169_pred['encoder.features.denseblock4.denselayer1.conv1.weight'])
    # print(para_ml_dilated_dense169_pred['encoder.features.denseblock4_branch2.denselayer1.conv1.weight'])
    # print(para_ml_dilated_dense169_pred['encoder.features.denseblock4_branch3.denselayer1.conv1.weight'])
    # print(para_ml_dense169_pred['encoder.features.denseblock4.denselayer1.conv1.weight'])
    # print(para_dilated_dense169_pred['encoder.features.denseblock4.denselayer1.conv1.weight'])
    # print(para_ml_dilated_dense169_pred['encoder.integ3.norm0.weight'])
    # print(para_ml_dilated_dense169_pool_pred['encoder.integ4.norm1.weight'])

    # test ml_dilated_densenet169_pred weights load on cat2000:
    # for key_, v_ in para_ml_dilated_dense169_pred.items():
    #     if torch.equal(para_ml_dilated_dense169_pred[key_], para_ml_dilated_dense169_pred_1[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    
    # test densenet169_pred weights load:
    # for key_, v_ in para_dense169.items():
    #     if torch.equal(para_dense169_pred['encoder.{}'.format(key_)], para_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    # # test dilated_densenet169_pred weights load:
    # for key_, v_ in para_dense169.items():
    #     if torch.equal(para_dilated_dense169_pred['encoder.{}'.format(key_)], para_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    # # test dilated_densenet169_pool_pred weights load:
    # for key_, v_ in para_dense169.items():
    #     if torch.equal(para_dilated_dense169_pool_pred['encoder.{}'.format(key_)], para_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    #
    # # test ml_densenet169_pred weights load:
    # for key_, v_ in para_dense169.items():
    #     if torch.equal(para_ml_dense169_pred['encoder.{}'.format(key_)], para_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    # # test ml_dilated_densenet169_pred weights load:
    # for key_, v_ in para_dense169.items():
    #     if torch.equal(para_ml_dilated_dense169_pred['encoder.{}'.format(key_)], para_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    # # test ml_dilated_densenet169_pool_pred weights load:
    # for key_, v_ in para_dense169.items():
    #     if torch.equal(para_ml_dilated_dense169_pool_pred['encoder.{}'.format(key_)], para_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))

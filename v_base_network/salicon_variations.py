import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict
from salicon_variations_config import *


__all__ = ['TwoBlockDenseNet', 'DilatedTwoBlockDenseNet', 'ThreeBlockDenseNet', 'DilatedThreeBlockDenseNet',
           'LowHighDenseNet', 'MiddleHighDenseNet', 'PlainDenseNet', 'plain_dense169_pred',
           'two_block_dense169_pred', 'two_dilated_block_dense169_pred', 'three_block_dense169_pred',
           'three_dilated_block_dense169_pred', 'low_high_concat_dense169_pred', 'middle_high_concat_dense169_pred']


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


class _TransitionWithoutConv(nn.Sequential):
    def __init__(self, num_input_features):
        super(_TransitionWithoutConv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
        #                                   kernel_size=1, stride=1, bias=False))
        # self.add_module('pad', nn.ZeroPad2d((1, 0, 0, 1)))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=1))


class _TransitionWithoutPool(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionWithoutPool, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class PlainDenseNet(nn.Sequential):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(PlainDenseNet, self).__init__()

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
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))


class TwoBlockDenseNet(nn.Sequential):

    def __init__(self, growth_rate=32, block_config=(6, 12),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(TwoBlockDenseNet, self).__init__()

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
                trans = _TransitionWithoutConv(num_input_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)


class DilatedTwoBlockDenseNet(nn.Sequential):

    def __init__(self, growth_rate=32, block_config=(6, 12),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DilatedTwoBlockDenseNet, self).__init__()

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
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=2)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _TransitionWithoutConv(num_input_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)


class ThreeBlockDenseNet(nn.Sequential):

    def __init__(self, growth_rate=32, block_config=(6, 12, 32),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(ThreeBlockDenseNet, self).__init__()

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
            else:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _TransitionWithoutConv(num_input_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)


class DilatedThreeBlockDenseNet(nn.Sequential):

    def __init__(self, growth_rate=32, block_config=(6, 12, 32),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DilatedThreeBlockDenseNet, self).__init__()

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
            else:
                block = _DilatedDenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate, dilated_rate=2)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                trans = _TransitionWithoutConv(num_input_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)


class LowHighDenseNet(nn.Sequential):

    def __init__(self, growth_rate=32, block_config=(6, 12, 32, 32),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(LowHighDenseNet, self).__init__()

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
        # num_block3_features = num_init_features
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
                # num_block3_features = num_features
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

        out = torch.cat((integ2, denseblock4), 1)

        return out


class MiddleHighDenseNet(nn.Sequential):

    def __init__(self, growth_rate=32, block_config=(6, 12, 32, 32),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(MiddleHighDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # num_block2_features = num_init_features
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
                # num_block2_features = num_features
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

        self.integ3 = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_block3_features)),
            ('relu0', nn.ReLU(inplace=True))]))

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

        integ3 = self.integ3(denseblock3)

        out = torch.cat((integ3, denseblock4), 1)

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


def plain_dense169(pretrained=False, **kwargs):
    model = PlainDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
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


def two_block_dense169(pretrained=False, **kwargs):
    model = TwoBlockDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12),
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


def two_dilated_block_dense169(pretrained=False, **kwargs):
    model = DilatedTwoBlockDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12),
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


def three_block_dense169(pretrained=False, **kwargs):
    model = ThreeBlockDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32),
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


def three_dilated_block_dense169(pretrained=False, **kwargs):
    model = DilatedThreeBlockDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32),
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


def low_high_concat_dense169(pretrained=False, **kwargs):
    model = LowHighDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
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


def middle_high_concat_dense169(pretrained=False, **kwargs):
    model = MiddleHighDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
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


def plain_dense169_pred(pretrained=False):
    model = DenseImageSalPre(plain_dense169, pretrained, 1664, 512, shape_out_)

    return model


def two_block_dense169_pred(pretrained=False):
    model = DenseImageSalPre(two_block_dense169, pretrained, 512, 512, shape_out_)

    return model


def two_dilated_block_dense169_pred(pretrained=False):
    model = DenseImageSalPre(two_dilated_block_dense169, pretrained, 512, 512, shape_out_)

    return model


def three_block_dense169_pred(pretrained=False):
    model = DenseImageSalPre(three_block_dense169, pretrained, 1280, 512, shape_out_)

    return model


def three_dilated_block_dense169_pred(pretrained=False):
    model = DenseImageSalPre(three_dilated_block_dense169, pretrained, 1280, 512, shape_out_)

    return model


def low_high_concat_dense169_pred(pretrained=False):
    model = DenseImageSalPre(low_high_concat_dense169, pretrained, 2176, 512, shape_out_)

    return model


def middle_high_concat_dense169_pred(pretrained=False):
    model = DenseImageSalPre(middle_high_concat_dense169, pretrained, 2944, 512, shape_out_)

    return model


if __name__ == '__main__':
    # Check weights load and feature size:
    # base network densenet-169
    model_plain_dense169_pred = plain_dense169_pred(pretrained=True)
    # summary(model_plain_dense169_pred, (3, 480, 640))

    model_two_block_dense169_pred = two_block_dense169_pred(pretrained=True)
    # summary(model_two_block_dense169_pred, (3, 480, 640))

    model_three_block_dense169_pred = three_block_dense169_pred(pretrained=True)
    # summary(model_three_block_dense169_pred, (3, 480, 640))

    model_two_dilated_block_dense169_pred = two_dilated_block_dense169_pred(pretrained=True)
    # summary(model_two_dilated_block_dense169_pred, (3, 480, 640))

    model_three_dilated_block_dense169_pred = three_dilated_block_dense169_pred(pretrained=True)
    # summary(model_three_dilated_block_dense169_pred, (3, 480, 640))

    model_lh_concat_dense169_pred = low_high_concat_dense169_pred(pretrained=True)
    # summary(model_lh_concat_dense169_pred, (3, 480, 640))

    model_mh_concat_dense169_pred = middle_high_concat_dense169_pred(pretrained=True)
    # summary(model_mh_concat_dense169_pred, (3, 480, 640))

    # base parameters
    para_plain_dense169 = model_plain_dense169_pred.state_dict()

    para_two_block_dense169 = model_two_block_dense169_pred.state_dict()
    para_two_dilated_block_dense169 = model_two_dilated_block_dense169_pred.state_dict()

    para_three_block_dense169 = model_three_block_dense169_pred.state_dict()
    para_three_dilated_block_dense169 = model_three_dilated_block_dense169_pred.state_dict()

    para_lh_concat_dense169 = model_lh_concat_dense169_pred.state_dict()
    para_mh_concat_dense169 = model_mh_concat_dense169_pred.state_dict()

    #  test weights load:
    # print('{} and {} value\n'.format('plain', 'two_block'))
    # for key_, v_ in para_two_block_dense169.items():
    #     if torch.equal(para_plain_dense169[key_], para_two_block_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    #
    # print('{} and {} value\n'.format('two_block', 'two_dilated_block'))
    # for key_, v_ in para_two_dilated_block_dense169.items():
    #     if torch.equal(para_two_block_dense169[key_], para_two_dilated_block_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    #
    # print('{} and {} value\n'.format('plain', 'three_block'))
    # for key_, v_ in para_three_block_dense169.items():
    #     if torch.equal(para_plain_dense169[key_], para_three_block_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))
    #
    # print('{} and {} value\n'.format('three_block', 'three_dilated_block'))
    # for key_, v_ in para_three_block_dense169.items():
    #     if torch.equal(para_three_block_dense169[key_], para_three_dilated_block_dense169[key_]):
    #         print('{}: equal'.format(key_))
    #     else:
    #         print('{}: no equal'.format(key_))

    print('{} and {} value\n'.format('plain', '2_4'))
    for key_, v_ in para_plain_dense169.items():
        if torch.equal(para_plain_dense169[key_], para_lh_concat_dense169[key_]):
            print('{}: equal'.format(key_))
        else:
            print('{}: no equal'.format(key_))

    print('{} and {} value\n'.format('plain', '4_4'))
    for key_, v_ in para_plain_dense169.items():
        if torch.equal(para_plain_dense169[key_], para_mh_concat_dense169[key_]):
            print('{}: equal'.format(key_))
        else:
            print('{}: no equal'.format(key_))




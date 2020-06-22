import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from salicon_config import *


__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet50_pred', 'resnet101_pred',
           'dilated_resnet50_pred', 'dilated_resnet101_pred', 'ml_dilated_resnet50_pred']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def dilated_conv3x3(in_planes, out_planes, stride=1, dilated=(1, 1)):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilated,
                     padding=dilated, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DilatedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilated=(1, 1), downsample=None):
        super(DilatedBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = dilated_conv3x3(planes, planes, stride, dilated)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DilatedResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(DilatedResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilated=(2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilated=(4, 4))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilated=(1, 1)):

        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, dilated, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilated=dilated))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class MutiscaleDilatedResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(MutiscaleDilatedResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilated=(2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilated=(4, 4))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilated=(1, 1)):

        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, dilated, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilated=dilated))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = torch.cat((x2, x3, x4), 1)

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


class ImageSalPre(nn.Module):

    def __init__(self, encode_model, pretrained, num_in_features, num_out_features, shape_out):
        super(ImageSalPre, self).__init__()
        # factor = Bottleneck.expansion
        self.encoder = encode_model(pretrained)
        self.predictor = Prediction(num_in_features, num_out_features, shape_out)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_model), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_model), strict=False)
    return model


def dilated_resnet50(pretrained=False, **kwargs):
    """Constructs a Dilated ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNet(DilatedBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_model), strict=False)
    return model


def dilated_resnet101(pretrained=False, **kwargs):
    """Constructs a Dilated ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNet(DilatedBottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_model), strict=False)
    return model


def ml_dilated_resnet50(pretrained=False, **kwargs):
    """Constructs a Dilated ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MutiscaleDilatedResNet(DilatedBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_model), strict=False)
    return model


def resnet50_pred(pretrained=False):
    """Constructs a ResNet-50 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ImageSalPre(resnet50, pretrained, 2048, 512, shape_out_)
    
    return model


def resnet101_pred(pretrained=False):
    """Constructs a ResNet-101 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ImageSalPre(resnet101, pretrained, 2048, 512, shape_out_)

    return model


def dilated_resnet50_pred(pretrained=False):
    """Constructs a ResNet-50 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ImageSalPre(dilated_resnet50, pretrained, 2048, 512, shape_out_)

    return model


def dilated_resnet101_pred(pretrained=False):
    """Constructs a ResNet-50 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ImageSalPre(dilated_resnet101, pretrained, 2048, 512, shape_out_)

    return model


def ml_dilated_resnet50_pred(pretrained=False):
    """Constructs a ResNet-50 model for Image Saliency Prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ImageSalPre(ml_dilated_resnet50, pretrained, 3584, 512, shape_out_)

    return model


if __name__ == '__main__':
    # Check weights load and feature size:
    model_res50_pred = resnet50_pred(pretrained=True)
    model_dilated_res50_pred = dilated_resnet50_pred(pretrained=True)
    model_ml_dilated_res50_pred = ml_dilated_resnet50_pred(pretrained=True)
    model_res50 = resnet50(pretrained=True)

    # para_res50_pred = model_res50_pred.state_dict()
    # para_res50 = model_res50.state_dict()
    # print(para_res50_pred['predictor.conv1_pred.weight'])
    # print(para_res50_pred['predictor.conv1_pred.bias'])
    # print(para_res50_pred['predictor.conv2_pred.weight'])
    # # test resnet50 weights load
    # for key, v in para_res50.items():
    #     if torch.equal(para_res50_pred['encoder.{}'.format(key)], para_res50[key]):
    #         print('{}: equal'.format(key))
    #     else:
    #         print('{}: no equal'.format(key))

    # check feature size
    # summary(model_res50_pred, (3, 480, 640))
    # summary(model_dilated_res50_pred, (3, 480, 640))
    summary(model_ml_dilated_res50_pred, (3, 480, 640))



import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


bn_momentum = 0.001


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.bn1 = norm_layer(in_planes, momentum=bn_momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(out_planes, momentum=bn_momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False, norm_layer=nn.BatchNorm2d):
        super(NetworkBlock, self).__init__()
        self.norm_layer = norm_layer
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual, self.norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0, num_splits=1):
        super(WideResNet, self).__init__()

        norm_layer = nn.BatchNorm2d if num_splits == 1 else partial(SplitBatchNorm, num_splits=num_splits)

        self.depth = depth

        n = 4
        if depth == 28:
            channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        else:
            channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 128*widen_factor]

        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True, norm_layer=norm_layer)
        # 2nd block
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate, norm_layer=norm_layer)
        # 3rd block
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate, norm_layer=norm_layer)

        if depth > 28:
            self.block4 = NetworkBlock(n, channels[3], channels[4], block, 2, drop_rate, norm_layer=norm_layer)

        # global average pooling and classifier
        self.bn1 = norm_layer(channels[-1], momentum=bn_momentum)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.channels = channels[-1]
        self.out_dim = channels[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        if self.depth > 28:
            out = self.block4(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return out


def build_wideresnet(depth, widen_factor, dropout, num_splits):
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,num_splits=num_splits)


def wide_resnet28w2(num_splits=1):
    encoder = build_wideresnet(28, 2, dropout=0.0, num_splits=num_splits)
    return encoder


def wide_resnet28w8(num_splits=1):
    encoder = build_wideresnet(28, 8, dropout=0.0, num_splits=num_splits)
    return encoder


def wide_resnet37w2(num_splits=1):
    encoder = build_wideresnet(37, 2, dropout=0.0, num_splits=num_splits)
    return encoder

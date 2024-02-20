"""
This is a 'simpler' version of `model.py` from [Person_Reidentification...](https://github.com/)
"""
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # For old pytorch, you may use kaiming_normal.
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|


class ClassBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 class_num,
                 droprate,
                 relu=False,
                 bnorm=True,
                 linear=512,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class FtNet(nn.Module):
    """The ResNet50-based Model"""

    def __init__(
        self,
        class_num=751, droprate=0.5, stride=2,
        circle=False, ibn=False, linear_num=512
    ):
        super(FtNet, self).__init__()

        resnet_weights = models.ResNet50_Weights.DEFAULT
        model_ft = models.resnet50(weights=resnet_weights)

        if ibn is True:
            model_ft = torch.hub.load(
                'XingangPan/IBN-Net',
                'resnet50_ibn_a',
                pretrained=True
            )

        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(
            2048, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class FtNetDense(nn.Module):
    """Define the DenseNet121-based Model"""

    def __init__(
        self,
        class_num, droprate=0.5, stride=2,
        circle=False, linear_num=512
    ):
        super().__init__()
        dense_weight = models.DenseNet121_Weights.DEFAULT
        model_ft = models.densenet121(weights=dense_weight)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()

        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        self.model = model_ft
        self.circle = circle

        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(
            1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        resnet_weights = models.ResNet50_Weights.DEFAULT
        model_ft = models.resnet50(weights=resnet_weights)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # remove the final down-sample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(
                self,
                name,
                ClassBlock(
                    2048,
                    class_num,
                    droprate=0.5,
                    linear=256,
                    relu=False,
                    bnorm=True
                )
            )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batch-size * 2048 * 6
        for i in range(self.part):
            part[i] = x[:, :, i].view(x.size(0), x.size(1))
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))

        # remove the final down-sample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y

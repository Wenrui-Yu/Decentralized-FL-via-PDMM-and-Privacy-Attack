import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.utils.data import Subset,Dataset
import torchvision
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import BasicBlock

# class Net(nn.Module):
#     def __init__(self, input_dim=32*32*3, hidden_dim=100, output_dim=10):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         out = F.relu(self.fc1(x))
#         # out=F.leaky_relu(self.fc1(x))
#         # out = torch.sigmoid(self.fc1(x))
#         out = self.fc2(out)
#         out = F.log_softmax(out, dim=1)
#         # out=F.softmax(out,dim=-1)
#         return out

# class Net(nn.Module):
#     def __init__(self, input_dim=32*32*3, hidden_dim=200, output_dim=100):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         out = F.relu(self.fc1(x))
#         # out=F.leaky_relu(self.fc1(x))
#         # out = torch.sigmoid(self.fc1(x))
#         out = self.fc2(out)
#         out = F.log_softmax(out, dim=1)
#         # out=F.softmax(out,dim=-1)
#         return out


# class Net(torchvision.models.ResNet):
#     """ResNet generalization for CIFAR thingies."""
#     # elif model == 'ResNet20':
#     #     model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16)
#     # elif model == 'ResNet32':
#     #     model = ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16)
#     def __init__(self, block=torchvision.models.resnet.BasicBlock, layers=[3,3,3], num_classes=100, zero_init_residual=False,
#                  groups=1, base_width=16, replace_stride_with_dilation=None,
#                  norm_layer=None, strides=[1, 2, 2, 2], pool='avg'):
#         """Initialize as usual. Layers and strides are scriptable."""
#         super(torchvision.models.ResNet, self).__init__()  # nn.Module
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer


#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False, False]
#         if len(replace_stride_with_dilation) != 4:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups

#         self.inplanes = base_width
#         self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)

#         self.layers = torch.nn.ModuleList()
#         width = self.inplanes
#         for idx, layer in enumerate(layers):
#             self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
#             width *= 2

#         self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
#         self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)


#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         for layer in self.layers:
#             x = layer(x)

#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x


## vgg
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_class=200):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


# #alexnet
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=96, out_channels=256, kernel_size=2, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Flatten(),
#             nn.Linear(in_features=256*6*6,out_features=4096),
#             nn.ReLU(),
#             nn.Linear(in_features=4096, out_features=256),
#             nn.ReLU(),
#             nn.Linear(in_features=256, out_features=200),
#             nn.Softmax()
#         )
#
#
#     def forward(self,x):
#         return self.model(x)
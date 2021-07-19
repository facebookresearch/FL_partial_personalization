import torch
import torch.nn as nn
from typing import Optional
from torchinfo import summary


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:
        super(ResidualBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.use_adapter = False
        self.adapter1 = False
        self.adapter2 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        if self.use_adapter:
            out = out + self.adapter1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_adapter:
            out = out + self.adapter2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def add_adapters(self):
        if not self.use_adapter:
            self.use_adapter = True
            self.adapter1 = AdapterBlock(self.planes)
            self.adapter2 = AdapterBlock(self.planes)

class AdapterBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.conv = conv1x1(planes, planes)  # 1x1 convolution
        # initialize
        nn.init.normal_(self.conv.weight, 0, 1e-4)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        identity = x
        out = self.bn(x)  # Batch norm
        out = self.conv(out)  # 1x1 conv
        out += identity  # skip connection
        return out

class EmnistResNet(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), num_classes=62):
        super().__init__()
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = torch.nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x 

    def print_summary(self, train_batch_size):
        device = next(self.parameters()).device
        print(summary(self, input_size=(train_batch_size, 1, 28, 28), device=device))

    def load_pretrained_and_prepare(self, state_dict, train_mode, layers_to_finetune, **kwargs):
        assert train_mode in ['train', 'finetune', 'finetune_res_layer', 
                              'finetune_inp_layer', 'finetune_out_layer',
                              'adapter'] 
        # load state_dict 
        self.load_state_dict(state_dict, strict=False)

        # Prepare
        if layers_to_finetune is None:  # do not fine tune
            layers_to_finetune = []
        if train_mode == 'finetune_res_layer' and len(layers_to_finetune) is None:
            raise ValueError(f'No residual blocks to finetune. Nothing to do')
        
        # Set requires_grad based on `train_mode`
        if train_mode in ['train', 'finetune']:  # all parameters are to be tuned
            def do_finetune(name):
                return True
        elif 'finetune_res_layer' in train_mode:
            # Finetune a specific residual block (available layers are [1, 2, 3, 4])
            def do_finetune(name):
                return any([f'layer{i}' in name for i in layers_to_finetune])
        elif train_mode in ['finetune_inp_layer']:
            # Fine tune positional and word embeddings
            def do_finetune(name):
                return (name in ['conv1.weight', 'bn1.weight', 'bn1.bias'])  # first conv + bn
        elif train_mode in ['finetune_out_layer']:
            # Fine tune final linear layer
            def do_finetune(name):
                return (name in ['fc.weight', 'fc.bias'])  # final fc
        elif train_mode in ['adapter']:
            # Train adapter modules (+ batch norm)
            def do_finetune(name):
                return ('adapter' in name) or ('bn1' in name) or ('bn2' in name)
            # Add adapter modules
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer.children():
                    # each block is of type `ResidualBlock`
                    block.add_adapters()
        else:
            raise ValueError(f'Unknown train_mode: {train_mode}')

        # set requires_grad for those parameters which need to be modified
        for name, param in self.named_parameters():
            param.requires_grad_(do_finetune(name))

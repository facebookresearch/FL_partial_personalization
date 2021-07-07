import torch
from torchvision.models import resnet18

def get_emnist_resnet():
    model = resnet18(num_classes=62)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model

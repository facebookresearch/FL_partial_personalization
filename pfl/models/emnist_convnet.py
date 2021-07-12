from collections import OrderedDict
import torch
from torchinfo import summary


class EmnistConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = torch.nn.Sequential(OrderedDict([
            ('c1', torch.nn.Conv2d(1, 32, kernel_size=(5, 5))),
            ('relu1', torch.nn.ReLU()),
            ('s2', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', torch.nn.Conv2d(32, 64, kernel_size=(5, 5))),
            ('relu3', torch.nn.ReLU()),
            ('s4', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        ]))
        self.fc = torch.nn.Linear(1024, 62)

    def forward(self, img_batch):
        output = self.convnet(img_batch)
        output = output.view(img_batch.shape[0], -1)
        output = self.fc(output)
        return output
    
    def print_summary(self, train_batch_size):
        device = next(self.parameters()).device
        print(summary(self, input_size=(train_batch_size, 1, 28, 28), device=device))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-26
# @Author  : Jehovah
# @File    : vggface.py
# @Software: PyCharm


import torch
import torch.nn.functional as F
import torch.utils.model_zoo

MODEL_URL = "https://github.com/claudio-unipv/vggface-pytorch/releases/download/v0.1/vggface-9d491dd7c30312.pth"

# It was 93.5940, 104.7624, 129.1863 before dividing by 255
MEAN_RGB = [
    0.367035294117647,
    0.41083294117647057,
    0.5066129411764705
]


def vggface(pretrained=False, **kwargs):
    """VGGFace model.

    Args:
        pretrained (bool): If True, returns pre-trained model
    """
    model = VggFace(**kwargs)
    if pretrained:
        state = torch.utils.model_zoo.load_url(MODEL_URL)
        model.load_state_dict(state)
    return model


class VggFace(torch.nn.Module):
    def __init__(self, classes=2622):
        """VGGFace model.

        Face recognition network.  It takes as input a Bx3x224x224
        batch of face images and gives as output a BxC score vector
        (C is the number of identities).
        Input images need to be scaled in the 0-1 range and then
        normalized with respect to the mean RGB used during training.

        Args:
            classes (int): number of identities recognized by the
            network

        """
        super(VggFace).__init__()
        self.conv1 = _ConvBlock(3, 64, 64)
        self.conv2 = _ConvBlock(64, 128, 128)
        self.conv3 = _ConvBlock(128, 256, 256, 256)
        self.conv4 = _ConvBlock(256, 512, 512, 512)
        self.conv5 = _ConvBlock(512, 512, 512, 512)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class _ConvBlock(torch.nn.Module):
    """A Convolutional block."""

    def __init__(self, *units):
        """Create a block with len(units) - 1 convolutions.

        convolution number i transforms the number of channels from
        units[i - 1] to units[i] channels.

        """
        super(_ConvBlock).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_, out, 3, 1, 1)
            for in_, out in zip(units[:-1], units[1:])
        ])

    def forward(self, x):
        # Each convolution is followed by a ReLU, then the block is
        # concluded by a max pooling.
        for c in self.convs:
            x = F.relu(c(x))
        return F.max_pool2d(x, 2, 2, 0, ceil_mode=True)


def _test_image(net, names, im):
    import torchvision
    tr = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MEAN_RGB, (1, 1, 1))
    ])
    x = tr(im)[None, ...]
    x = net(x)
    y = torch.nn.functional.softmax(x, 1)
    N = 5
    rank = torch.topk(y[0, :], N)
    for i in range(N):
        index = rank[1][i].item()
        score = rank[0][i].item()
        print("{}) {} ({:.2f})".format(i + 1, names[index], score))


def _test():
    import sys
    from PIL import Image
    net = vggface(True)
    net.eval()
    names = open("names.txt").read().split()
    with torch.no_grad():
        for path in sys.argv[1:]:
            print(path)
            _test_image(net, names, Image.open(path))
            print()


if __name__ == "__main__":
    _test()
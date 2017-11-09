import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable


def conv3x3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv3x3x3up(in_planes, out_planes, bias=True):
    """3x3x3 convolution with padding"""
    return nn.ConvTranspose3d(
        in_planes, out_planes, stride=2, kernel_size=3, padding=1, output_padding=1
    )


def convbn(in_planes, out_planes, stride=1, bias=True):
    return nn.Sequential(
        (conv3x3x3(in_planes, out_planes, stride=stride, bias=bias)),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True),
    )


def convbnup(in_planes, out_planes, bias=True):
    return nn.Sequential(
        (conv3x3x3up(in_planes, out_planes, bias=bias)),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    def __init__(self, opts):
        super(Net, self).__init__()
        try:
            embedding_dim = opts.embedding_dim
        except:
            embedding_dim = 8
        try:
            num_words = opts.num_words
        except:
            num_words = 3
        try:
            num_classes = opts.nclasses
        except:
            num_classes = 18
        try:
            num_layers = opts.num_layers
        except:
            num_layers = 4  # 32x32x32 input
        try:
            hidden_dim = opts.hdim
        except:
            hidden_dim = 64

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.layers = nn.ModuleList()
        self.upscales = nn.ModuleList()
        self.outmaps = nn.ModuleList()
        indim = embedding_dim
        outdim = hidden_dim
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = nn.Sequential(convbn(indim, outdim), convbn(outdim, outdim, stride=2))
            indim = outdim
            self.layers.append(layer)
            if i > 0:
                self.upscales.append(convbnup(outdim, outdim))
            if i < num_layers - 1:
                to_output = nn.Sequential(
                    convbn(2 * outdim, outdim), nn.Conv3d(outdim, num_classes, kernel_size=1)
                )
            else:
                to_output = nn.Conv3d(outdim, num_classes, kernel_size=1)

            self.outmaps.append(to_output)

    def forward(self, x):
        # FIXME for pytorch .4: embedding is better now
        szs = list(x.size())
        x = x.view(-1)
        z = self.embedding(x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.transpose(1, 4).contiguous()
        temps = []

        for i in range(self.num_layers):
            z = self.layers[i](z)
            temps.append(z)

        for i in range(self.num_layers - 1):
            temps[i] = torch.cat((temps[i], self.upscales[i](temps[i + 1])), 1)

        outs = []
        for i in range(self.num_layers):
            outs.append(self.outmaps[i](temps[i]))
        return outs


class multiscale_bcewl(nn.Module):
    def __init__(self, weights=None):
        super(multiscale_bcewl, self).__init__()
        if weights is None:
            weights = [1, 1, 1, 1]
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inp, target):
        loss = 0
        for i in range(4):
            y = inp[i]
            y = y.transpose(1, 4).contiguous()
            loss += self.weights[i] * self.bce(y, target[i])
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=18, help="num_classes")
    parser.add_argument("--num_words", type=int, default=3, help="number of words in embedding")
    parser.add_argument("--imsize", type=int, default=64, help="imsize, use 32 or 64")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--hsize", type=int, default=64, help="hidden dim")
    opts = parser.parse_args()

    net = Net(opts)
    x = torch.LongTensor(7, 32, 32, 32).zero_()
    y = net(Variable(x))

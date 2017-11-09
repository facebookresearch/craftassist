import argparse
import torch
import torch.nn as nn


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
            num_layers = opts.num_layers
        except:
            num_layers = 4  # 32x32x32 input
        try:
            hidden_dim = opts.hidden_dim
        except:
            hidden_dim = 64

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.layers = nn.ModuleList()
        indim = embedding_dim
        outdim = hidden_dim
        self.num_layers = num_layers
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(indim, outdim, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm3d(outdim),
                nn.ReLU(inplace=True),
            )
        )
        indim = outdim
        for i in range(num_layers - 1):
            layer = nn.Sequential(convbn(indim, outdim), convbn(outdim, outdim, stride=2))
            indim = outdim
            self.layers.append(layer)
        self.out = nn.Linear(outdim, 1)

        # todo normalize things?  margin doesn't mean much here

    def forward(self, x):
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        szs = list(x.size())
        x = x.view(-1)
        z = self.embedding.weight.index_select(0, x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(self.num_layers):
            z = self.layers[i](z)
        z = z.mean([2, 3, 4])
        #        szs = list(z.size())
        #        z = z.view(szs[0], szs[1], -1)
        #        z = z.max(2)[0]
        #        z = nn.functional.normalize(z, dim=1)
        return self.out(z)


class rank_loss(nn.Module):
    def __init__(self, margin=0.1, nneg=5):
        super(rank_loss, self).__init__()
        self.nneg = 5
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, inp):
        # it is expected that the batch is arranged as pos neg neg ... neg pos neg ...
        # with self.nneg negs per pos
        assert inp.shape[0] % (self.nneg + 1) == 0
        inp = inp.view(self.nneg + 1, -1)
        pos = inp[0]
        neg = inp[1:].contiguous()
        errors = self.relu(neg - pos.repeat(self.nneg, 1) + self.margin)
        return errors.mean()


class reshape_nll(nn.Module):
    def __init__(self, nneg=5):
        super(reshape_nll, self).__init__()
        self.nneg = nneg
        self.lsm = nn.LogSoftmax()
        self.crit = nn.NLLLoss()

    def forward(self, inp):
        # it is expected that the batch is arranged as pos neg neg ... neg pos neg ...
        # with self.nneg negs per pos
        assert inp.shape[0] % (self.nneg + 1) == 0
        inp = inp.view(-1, self.nneg + 1).contiguous()
        logsuminp = self.lsm(inp)
        o = torch.zeros(inp.size(0), device=inp.device).long()
        return self.crit(logsuminp, o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_words", type=int, default=3, help="number of words in embedding")
    parser.add_argument("--imsize", type=int, default=32, help="imsize, use 32 or 64")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--hsize", type=int, default=64, help="hidden dim")
    opts = parser.parse_args()

    net = Net(opts)
    x = torch.LongTensor(7, 32, 32, 32).zero_()
    y = net(x)

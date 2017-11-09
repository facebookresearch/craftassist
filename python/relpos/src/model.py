import torch
import torch.nn as nn

# from config import DEBUG

""" Convolution Block: conv, batch norm, relu """


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


""" Baseline Neural Network Model: 1 local box -> CNN -> score """


class BaselineNN(torch.nn.Module):
    def __init__(self, config):
        super(BaselineNN, self).__init__()
        self.config = config
        self.bsize = self.config.args.boxsize
        self.vsize = self.config.vsize
        self.h_dim = self.config.args.h_dim
        self.dropout_rate = self.config.args.dropout_rate

        # TODO(demiguo): think more about the model
        assert self.bsize % 3 == 0  # current model assumption
        self.conv_input = ConvBlock(self.vsize, self.h_dim, 3)

        self.bsize1 = self.bsize
        self.conv1 = ConvBlock(self.h_dim, self.h_dim, 3)
        self.max1 = nn.MaxPool3d(3)

        self.bsize2 = self.bsize // 3
        self.conv2 = ConvBlock(self.h_dim, self.h_dim, 3)

        self.bsize3 = self.bsize2
        self.final_layer = nn.Linear(self.h_dim * (self.bsize3 ** 3), self.h_dim)
        self.score_layer = nn.Linear(self.h_dim, 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def get_train_parameters(self):
        params = []
        total = 0
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
                total += param.numel()
        self.config.log.info("total parameters: %d" % total)
        return params

    def one_hot_embedding(self, x):
        # x: LongTensor of [batch_size, N]
        # return LongTensor of [batch_size, N, vsize]
        assert x.dim() == 2
        one_hot_x = torch.zeros(x.size(0), x.size(1), self.vsize, device=x.device)
        one_hot_x.scatter_(2, x.unsqueeze(2), 1)
        return one_hot_x

    def forward(self, box):
        self.batch_size = box.size(0)
        assert box.size() == (self.batch_size, self.bsize, self.bsize, self.bsize)

        # one hot embedding layer
        box = box.contiguous().view(self.batch_size, self.bsize ** 3)
        box_embeds = self.one_hot_embedding(box).view(
            self.batch_size, self.bsize, self.bsize, self.bsize, self.vsize
        )

        # encode layers

        box_input = box_embeds.permute(4, 0, 1, 2, 3).transpose(0, 1)
        assert box_input.size() == (
            self.batch_size,
            self.vsize,
            self.bsize,
            self.bsize,
            self.bsize,
        )
        box_out = self.conv_input(box_input)
        assert box_out.size() == (self.batch_size, self.h_dim, self.bsize, self.bsize, self.bsize)

        box_out = self.conv1(box_out)
        assert box_out.size() == (self.batch_size, self.h_dim, self.bsize, self.bsize, self.bsize)
        box_out = self.max1(box_out)
        assert box_out.size() == (
            self.batch_size,
            self.h_dim,
            self.bsize2,
            self.bsize2,
            self.bsize2,
        )

        box_out = self.conv2(box_out)
        assert box_out.size() == (
            self.batch_size,
            self.h_dim,
            self.bsize2,
            self.bsize2,
            self.bsize2,
        )

        # final layer
        box_out = box_out.contiguous().view(self.batch_size, self.h_dim * (self.bsize3 ** 3))
        box_out = self.final_layer(box_out)
        assert box_out.size() == (self.batch_size, self.h_dim)
        box_out = self.dropout(box_out)
        box_score = self.score_layer(box_out)
        assert box_score.size() == (self.batch_size, 1)
        box_score = nn.Sigmoid()(box_score)
        return box_score

    """ Get normalized ranker score """

    def get_score(self, box):
        return self(box)  # now, it's same as forward function

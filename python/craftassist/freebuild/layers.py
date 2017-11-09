import torch.nn as nn

from .utils import ZeroPad3d
from .utils import down_shift


class conv3d_fc(nn.Conv3d):
    def __init__(self, num_in, num_out):
        super(conv3d_fc, self).__init__(num_in, num_out, 1)


class down_shifted_conv3d(nn.Module):
    def __init__(
        self,
        num_filters_in,
        num_filters_out,
        filter_size=(3, 2, 3),
        stride=(1, 1, 1),
        shift_output_down=False,
        norm=None,
    ):
        super(down_shifted_conv3d, self).__init__()

        assert norm in [None, "batch_norm", "weight_norm"]
        # self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.conv = nn.Conv3d(num_filters_in, num_filters_out, filter_size, stride, bias=False)
        # self.conv.weight.data[:] = 0.
        # print(self.conv.weight.data.shape)
        # self.conv.weight.data[0,0,1,1,1] = 1.
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = ZeroPad3d(
            (
                int((filter_size[2] - 1) / 2),  # pad front left
                int((filter_size[2] - 1) / 2),  # pad back right
                filter_size[1] - 1,  # pad top
                0,  # pad down
                int((filter_size[0] - 1) / 2),  # pad left
                int((filter_size[0] - 1) / 2),
            )
        )  # pad right

        if norm == "weight_norm":
            pass
        elif norm == "batch_norm":
            self.bn = nn.BatchNorm3d(num_filters_out)

        if shift_output_down:
            self.down_shift = lambda x: down_shift(x)

    def forward(self, x):
        x = self.pad(x)
        # print('padding:', x.shape)
        x = self.conv(x)
        # print('conv:', x.shape)
        x = self.bn(x) if self.norm == "batch_norm" else x
        return self.down_shift(x) if self.shift_output_down else x


"""
class VoxelCNN(nn.Module):
    def __init__(self, in_dim=1, in_channels=1, out_dim=2):
        super(VoxelCNN, self).__init__()
        self.relu = nn.ReLU()

        # conv layers
        self.conv_init = down_shifted_conv3d(in_dim, in_channels)
        self.conv1 = down_shifted_conv3d(in_channels, in_channels)
        self.conv2 = down_shifted_conv3d(in_channels, in_channels)
        self.conv3 = down_shifted_conv3d(in_channels, in_channels)

    def forward(self, x):
        out = self.relu(self.conv_init(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        return out
"""

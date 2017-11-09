import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.append("..")
sys.path.append("../..")
from voxel_cnn_base import VoxelCNN
from voxel_cube import make_center
from mask_rcnn import compute_bce_f1


class Identity(nn.Module):
    def forward(self, x):
        return x


class LocalMaskExpander(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.expander = nn.Sequential(
            nn.Linear(C, C // 2),
            nn.ReLU(),
            nn.Linear(C // 2, C // 4),
            nn.ReLU(),
            nn.Linear(C // 4, 2 ** 3),
        )  # 2x2x2 expand

    def forward(self, feat_map, coarse_mask):
        """
        This function takes each 1's location on coarse_mask and its
        corresponding feature vector on feat_map, and uses an MLP to
        predict its 2x2x2 neighborhood in the finer scale
        """
        assert len(coarse_mask.size()) == 4  # B,H,W,D
        self.one_idx = torch.nonzero(coarse_mask)
        B = self.one_idx.size()[0]
        feat_map = feat_map.permute(0, 2, 3, 4, 1)  # put channel to last
        feat_vec = feat_map[self.one_idx.split(1, dim=1)]
        feat_vec = feat_vec.view(B, -1)
        return self.expander(feat_vec)

    def forward_and_loss(self, feat_map, coarse_mask, fine_mask):
        """
        Given a feature map and a groundtruth mask at a coarse scale, compute
        the refinement loss only for activated mask elements
        """
        assert coarse_mask.sum() > 0
        out = self(feat_map, coarse_mask)
        B = out.size()[0]
        target = []
        batch_idx = self.one_idx[:, 0]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x_idx = 2 * self.one_idx[:, 1] + i
                    y_idx = 2 * self.one_idx[:, 2] + j
                    z_idx = 2 * self.one_idx[:, 3] + k
                    select = fine_mask[(batch_idx, x_idx, y_idx, z_idx)]
                    target.append(select.view(B, -1))
        target = torch.cat(target, dim=-1)
        return nn.BCEWithLogitsLoss()(out, target)

    def predict(self, feat_map, coarse_mask):
        B, H, W, D = coarse_mask.size()
        fine_mask = torch.zeros(B, H * 2, W * 2, D * 2)
        if fine_mask.device != coarse_mask.device:
            fine_mask = fine_mask.to(coarse_mask.device)
        if coarse_mask.sum() > 0:
            out = (self(feat_map, coarse_mask) > 0).float()
            batch_idx = self.one_idx[:, 0]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        x_idx = 2 * self.one_idx[:, 1] + i
                        y_idx = 2 * self.one_idx[:, 2] + j
                        z_idx = 2 * self.one_idx[:, 3] + k
                        fine_mask[(batch_idx, x_idx, y_idx, z_idx)] = out[:, i * 4 + j * 2 + k]
        return fine_mask


class MaskModel(VoxelCNN):
    def __init__(self, tags=[], opts={}, model_path=None):
        self.n_scales = opts.get("n_scales", 3)
        super().__init__(opts, tags, model_path)

    def _init_net(self):
        voxel_embedding_dim = self.opts["voxel_embedding_dim"]
        self.embedding = nn.Embedding(200, voxel_embedding_dim)  # see random_shape_helpers.py

        strides = [2] * self.n_scales
        channels = [4 * voxel_embedding_dim] * self.n_scales
        in_channels = voxel_embedding_dim * 2

        self.layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.mask_layer = nn.ModuleList()
        self.refine_layers = nn.ModuleList()

        ###################
        # scale     size
        # 0         16
        # 1         8
        # 2         4
        # 3         2
        ###################
        for i in range(self.n_scales):
            module = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(),
                nn.Conv3d(in_channels, channels[i], kernel_size=3, stride=strides[i], padding=1),
                nn.BatchNorm3d(channels[i]),
                nn.ReLU(),
            )
            in_channels = channels[i]
            self.layers.append(module)

        for i in range(1, self.n_scales):
            if i == self.n_scales - 1:
                deconv_in_dim = channels[i]
            else:
                deconv_in_dim = channels[i] * 2
            self.up_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        deconv_in_dim,
                        channels[i - 1],
                        kernel_size=3,
                        stride=strides[i],
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm3d(channels[i - 1]),
                    nn.ReLU(),
                )
            )

        # we output the init mask at n_scales // 2 - 1
        self.mask_layer = nn.Conv3d(
            channels[self.n_scales // 2 - 1] * 2, 1, kernel_size=3, stride=1, padding=1
        )

        # then refine the mask
        for i in range(self.n_scales // 2):
            self.refine_layers.append(LocalMaskExpander(channels[i] * 2))

    def forward(self, x, mask):
        B, H, W, D = x.size()
        x = self.embedding(x.long())
        x = torch.cat(
            (x, mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.opts["voxel_embedding_dim"])), dim=-1
        )
        x = x.permute(0, 4, 1, 2, 3)  # B, C, H, W, D

        temps = []
        for i in range(self.n_scales):
            x = self.layers[i](x)
            temps.append(x)

        feat_maps = [None] * self.n_scales
        for i in range(self.n_scales - 1, 0, -1):
            if i == self.n_scales - 1:
                z = x
            else:
                z = torch.cat((x, temps[i]), dim=1)
            feat_maps[i] = z
            x = self.up_layers[i - 1](z)
        feat_maps[0] = torch.cat((x, temps[0]), dim=1)
        return feat_maps

    def forward_and_loss(self, x, mask, target):
        B = x.size()[0]
        targets = self.subsample_targets(target)
        feat_maps = self(x, mask)
        out_scale = self.n_scales // 2 - 1
        out_mask = self.mask_layer(feat_maps[out_scale])
        loss = []
        for i, rl in enumerate(self.refine_layers):
            loss.append(rl.forward_and_loss(feat_maps[i], targets[i + 1], targets[i]))
        loss.append(
            nn.BCEWithLogitsLoss()(out_mask.view(B, -1), targets[out_scale + 1].view(B, -1))
        )
        return loss

    def predict(self, x, mask):
        feat_maps = self(x, mask)
        out_mask = (self.mask_layer(feat_maps[self.n_scales // 2 - 1]) > 0).float()
        masks = [out_mask.squeeze(1)]  # remove the channel dim
        for i in range(len(self.refine_layers) - 1, -1, -1):
            masks.append(self.refine_layers[i].predict(feat_maps[i], masks[-1]))
        masks.reverse()
        return masks

    def train_epochs(
        self,
        training_data,
        validation_data,
        save_model_path,
        lr=1e-4,
        batch_size=64,
        max_epochs=100,
    ):
        """ """
        opt = optim.Adam(self.parameters(), lr=lr)

        training_data = [tuple([torch.from_numpy(x) for x in d]) for d in training_data]
        validation_data = [tuple([torch.from_numpy(x) for x in d]) for d in validation_data]

        for m in range(max_epochs):
            self.train()  ## because we do prediction after each epoch
            random.shuffle(training_data)
            loss = []

            xs, masks, targets = self.sample_an_epoch(training_data, 3)

            for i in range(0, xs.size()[0], batch_size):
                if i % batch_size == 0:
                    print(".", end="", flush=True)

                x = xs[i : i + batch_size]
                mask = masks[i : i + batch_size]
                target = targets[i : i + batch_size]

                if self.is_cuda():
                    x = x.cuda()
                    mask = mask.cuda()
                    target = target.cuda()

                opt.zero_grad()
                ls = self.forward_and_loss(x, mask, target)
                loss.append(ls)
                l = sum(ls)
                l.backward()
                opt.step()

            print("\nEpoch {}".format(m))
            for i, ls in enumerate(zip(*loss)):
                print("Resoluton 1/{} loss: {}".format(2 ** i, sum(ls) / len(ls)))
            print("Eval on training data:")
            self.validate((xs, masks, targets), batch_size)
            print("Eval on validation data:")
            self.validate(self.sample_an_epoch(validation_data, 3), batch_size)

        self.save(save_model_path)

    def subsample_targets(self, target):
        # target will be binary
        targets = [target]
        for i in range(self.n_scales):
            with torch.no_grad():
                t = nn.MaxPool3d(kernel_size=2, stride=2)(targets[-1].unsqueeze(1)).squeeze(1)
                targets.append(t)
        return targets

    def sample_an_epoch(self, data, samples_per_house):
        x, mask, target = [], [], []
        for obj, anno in data:
            x_, mask_, target_ = self.sample_minibatch(obj, anno, samples_per_house)
            if x_ is None:
                continue
            x.append(x_)
            mask.append(mask_)
            target.append(target_)
        return torch.cat(x, dim=0), torch.cat(mask, dim=0), torch.cat(target, dim=0)

    def sample_minibatch(self, obj, anno, batch_size):
        idx = torch.nonzero(anno > 0)
        if idx.nelement() == 0:
            return [None] * 3
        idx = idx[torch.randperm(idx.size()[0])]
        idx = idx[:batch_size]  # only sample a minibatch
        B = idx.size()[0]
        x = torch.stack([obj] * B, dim=0)
        mask = torch.zeros_like(x, dtype=torch.float32)
        idx = idx.split(1, dim=1)
        bidx = torch.arange(0, B).unsqueeze(-1)
        mask[(bidx.long(), idx[0], idx[1], idx[2])] = 1

        target = anno[idx]
        target = (
            torch.stack([anno] * B, dim=0).view(B, -1) == target.repeat(1, anno.nelement())
        ).float()
        return x, mask, target.view(mask.size())

    def validate(self, validation_data, batch_size):
        self.eval()

        xs, masks, targets = validation_data
        ps = []
        for i in range(0, xs.size()[0], batch_size):
            x = xs[i : i + batch_size]
            mask = masks[i : i + batch_size]
            target = targets[i : i + batch_size]

            if self.is_cuda():
                x = x.cuda()
                mask = mask.cuda()
                target = target.cuda()

            with torch.no_grad():
                outhats = self.predict(x, mask)
            ts = self.subsample_targets(target)[: len(outhats)]

            for batch_idx in range(x.size()[0]):
                ps.append(
                    [
                        compute_bce_f1(mask[batch_idx], target[batch_idx])
                        for mask, target in zip(outhats, ts)
                    ]
                )

        for i, f1 in enumerate(zip(*ps)):
            print("Resoluton 1/{} F1: {}".format(2 ** i, np.mean(f1)))


if __name__ == "__main__":
    with open("./training_data.pkl", "rb") as f:
        training_data = pickle.load(f)
    with open("./validation_data.pkl", "rb") as f:
        validation_data = pickle.load(f)

    size = -1
    for obj, _ in training_data:
        size = max(size, max(obj.shape))
    size += 1
    print("size: {}".format(size))

    training_data = [
        (make_center(td[0], size), make_center(td[1], size)) for td in training_data[:10000]
    ]
    validation_data = [
        (make_center(vd[0], size), make_center(vd[1], size)) for vd in validation_data[:1000]
    ]

    print("loading finished!")

    model = MaskModel(opts=dict(voxel_embedding_dim=24, n_scales=4, gpu_id=1))
    model.train_epochs(
        training_data, validation_data, "/tmp/model.pth", max_epochs=500, batch_size=64, lr=1e-3
    )

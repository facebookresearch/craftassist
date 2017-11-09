import torch
import torch.nn as nn
import numpy as np
import logging


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


def load_voxel_cnn(model_path, model_opts):
    sds = torch.load(model_path)
    model = sds["class"](model_path=model_path, opts=model_opts)
    return model


def precision_at(pred, out):
    p1, pk = {}, {}
    for p, o in zip(pred, out):
        if o not in p1:
            p1[o] = []
            pk[o] = []
        p1[o].append(float(p[0] == o))
        pk[o].append(float(o in p))

    for o in p1.keys():
        p1[o] = np.mean(p1[o])
        pk[o] = np.mean(pk[o])
    return np.mean(list(p1.values())), np.mean(list(pk.values()))


class VoxelCNN(nn.Module):
    """
    A base class for processing 3d voxels with convnets
    """

    def __init__(self, opts, tags, model_path):
        super().__init__()

        self.opts = dict(voxel_embedding_dim=16, hidden_dim=128, data_augmentation=True, gpu_id=0)

        if model_path:
            # load stored model options
            sds = torch.load(model_path)
            self.tags = sds["tags"]
            self.opts.update(sds["opts"])
        else:
            self.tags = tags

        self.opts.update(opts)

        self._init_net()

        # load stored parameters
        if model_path:
            self.load_state_dict(sds["model"])

        max_tags_n = max([n for t, n in tags])
        tag_weights = np.array([max_tags_n / float(n) for t, n in tags])
        self.tag_weights = torch.from_numpy(tag_weights)
        self.set_device()

        if self.opts["gpu_id"] >= 0:
            self.tag_weights = self.tag_weights.cuda()

    def set_device(self):
        gpu = self.opts["gpu_id"]
        if torch.cuda.is_available() and gpu >= 0:
            logging.info("Voxel model running on GPU:{}".format(gpu))
            torch.cuda.set_device(gpu)
            self.cuda()
        else:
            logging.info("Voxel model running on CPU")
            self.cpu()

    def _data_augmentation(self, cube, idx=None):
        """
        Randomly transform the cube for more data.
        The transformation is chosen from:
            0. original
            1. x-z plane rotation 90
            2. x-z plane rotation 180
            3. x-z plane rotation 270
            4. x-axis flip
            5. z-axis flip

        cube - [X, Y, Z, 1]
        """

        def transform(c, idx):
            idx = np.random.choice(range(6)) if (idx is None) else idx
            if idx == 0:
                c_ = c
            elif idx >= 1 and idx <= 3:  # rotate
                npc = c.cpu().numpy()
                npc = np.rot90(npc, idx, axes=(0, 2))  # rotate on the x-z plane
                c_ = torch.from_numpy(npc.copy()).to(c.device)
            else:  # flip
                npc = c.cpu().numpy()
                npc = np.flip(npc, axis=(idx - 4) * 2)  # 0 or 2
                c_ = torch.from_numpy(npc.copy()).to(c.device)
            return c_, idx

        X, Y, Z, _ = cube.shape
        assert X == Y and Y == Z, "This only applies to cubes!"
        cube, idx = transform(cube[:, :, :, 0], idx)
        return cube.unsqueeze(-1), idx

    def save(self, model_path):
        self.cpu()
        sds = {}
        sds["class"] = self.__class__
        sds["tags"] = self.tags
        sds["opts"] = self.opts
        sds["model"] = self.state_dict()
        torch.save(sds, model_path)
        self.set_device()

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def _init_net(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def train_epochs(
        self,
        training_data,
        validation_data,
        loss_fn,
        pred_fn,
        save_model_path,
        lr=1e-4,
        batch_size=64,
        max_epochs=100,
    ):
        raise NotImplementedError()

    def validate(self, validation_data, pred_fn, batch_size):
        raise NotImplementedError()

    def predict_object(self, np_blocks, batch_size):
        raise NotImplementedError()

import torch
import torch.nn as nn
import numpy as np
import mc_block_ids as mbi
import logging


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


def load_voxel_cnn(model_path, model_opts):
    sds = torch.load(model_path)
    model = sds["class"](model_path=model_path, opts=model_opts)
    return model


class VoxelCNN(nn.Module):
    """
    A base class for processing 3d voxels with convnets
    """

    def __init__(self, opts, tags, model_path):
        super(VoxelCNN, self).__init__()

        self.opts = dict(
            voxel_embedding_dim=16,
            hidden_dim=128,
            data_augmentation=True,
            voxel_group=True,
            gpu_id=0,
        )

        if model_path:
            # load stored model options
            sds = torch.load(model_path)
            self.tags = sds["tags"]
            self.opts.update(sds["opts"])
        else:
            assert tags
            self.tags = tags

        self.opts.update(opts)

        mbi.voxel_group = self.opts["voxel_group"]

        self._init_net()

        # load stored parameters
        if model_path:
            self.load_state_dict(sds["model"])

        self.set_device()

    def set_device(self):
        gpu = self.opts["gpu_id"]
        if torch.cuda.is_available() and gpu >= 0:
            logging.info("Voxel model running on GPU:{}".format(gpu))
            self.cuda(gpu)
        else:
            logging.info("Voxel model running on CPU")
            self.cpu()

    def _data_augmentation(self, cube):
        """
        Randomly transform the cube for more data.
        The transformation is chosen from:
            0. original
            1. x-z plane rotation 90
            2. x-z plane rotation 180
            3. x-z plane rotation 270
            4. x-axis flip
            5. z-axis flip

        cube - [X, Y, Z, 4]
        """

        def transform(c):
            idx = np.random.choice(range(6))
            if idx == 0:
                return c
            elif idx >= 1 and idx <= 3:  # rotate
                npc = c.cpu().numpy()
                npc = np.rot90(npc, idx, axes=(0, 2))  # rotate on the x-z plane
                return torch.from_numpy(np.ascontiguousarray(npc)).to(c.device)
            else:  # flip
                npc = c.cpu().numpy()
                npc = np.flip(npc, axis=(idx - 4) * 2)  # 0 or 2
                return torch.from_numpy(np.ascontiguousarray(npc)).to(c.device)

        X, Y, Z, _ = cube.shape
        assert X == Y and Y == Z, "This only applies to cubes!"
        return transform(cube[:, :, :, 0]).unsqueeze(-1)

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

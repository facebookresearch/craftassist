import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import logging
import mc_block_ids as mbi
from voxel_cnn_base import VoxelCNN, Flatten, precision_at
from voxel_cube import get_batch_voxel_center_cubes
from seg_crf_models import VoxelCRF


class LocalVoxelCNN(VoxelCNN):
    cube_size = 9
    """
    A model for classifying a voxel fully depending on its local 9x9x9 neighborhood
    This is the v0 vision model.
    """

    def __init__(self, tags=[], opts={}, model_path=None):
        super().__init__(opts, tags, model_path)
        if model_path is not None:
            # CRF hyperparameters are set empirically
            self.crf = VoxelCRF(
                pairwise_weight1=30.0, sxyz1=2.0, sembed=(0.5 if mbi.voxel_group else 3.0)
            )
        else:
            self.crf = VoxelCRF()

    def _init_net(self):
        """
        See a brief description of the method at
         https://fb.quip.com/eCdkAK9cHZoC#LKdACAnDQpA
        """
        num_tags = len(self.tags)
        voxel_embedding_dim = self.opts["voxel_embedding_dim"]
        self.input_size = self.cube_size
        hidden_dim = self.opts["hidden_dim"]

        print("Vocabulary size: {}".format(mbi.total_ids_n()))
        self.embedding = nn.Embedding(mbi.total_ids_n(), voxel_embedding_dim)

        kernel_sizes = [3, 3, 3, 3]
        strides = [1, 1, 1, 1]
        channels = [
            2 * voxel_embedding_dim,  # 7
            4 * voxel_embedding_dim,  # 5
            8 * voxel_embedding_dim,  # 3
            8 * voxel_embedding_dim,
        ]  # 1
        modules = []

        in_channels = voxel_embedding_dim
        s = self.input_size
        for i in range(len(kernel_sizes)):
            modules.append(
                nn.Conv3d(in_channels, channels[i], kernel_size=kernel_sizes[i], stride=strides[i])
            )
            modules.append(nn.BatchNorm3d(channels[i]))
            modules.append(nn.ReLU())
            in_channels = channels[i]
            s = (s - kernel_sizes[i]) // strides[i] + 1

        modules.append(Flatten())
        modules.append(nn.Linear(s ** 3 * channels[-1], hidden_dim))
        modules.append(nn.BatchNorm1d(hidden_dim))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.3))

        self.cnn = nn.Sequential(*modules)
        self.center_layer = nn.Sequential(
            nn.Linear(voxel_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, num_tags))

    def forward(self, x):
        _, H, W, D, C = x.size()
        assert H == self.input_size
        assert C == 1
        voxel_ids = x[:, :, :, :, 0].long()
        x = self.embedding(voxel_ids)
        ## extract the embedding for the center voxel
        ccx = x[:, H // 2, W // 2, D // 2, :]
        x = x.transpose(1, 4)  ## B,C,H,W,D
        return self.classifier(torch.cat([self.cnn(x), self.center_layer(ccx)], dim=-1))

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

        training_data_np = training_data
        training_data = [tuple([torch.from_numpy(x) for x in d]) for d in training_data]
        if self.is_cuda():
            training_data = [
                tuple([x.cuda(self.opts["gpu_id"]) for x in d]) for d in training_data
            ]

        for m in range(max_epochs):
            self.train()  ## because we do prediction after each epoch
            random.shuffle(training_data)
            loss = []
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i : i + batch_size]
                ipt, out = zip(*batch)
                if self.opts["data_augmentation"]:
                    ipt = [self._data_augmentation(cube)[0] for cube in ipt]
                ipt = torch.stack(ipt, dim=0)
                out = torch.stack(out, dim=0)
                outhat = self(ipt)
                opt.zero_grad()
                l = self._loss_fn(outhat, out)
                loss.append(l)
                l.backward()
                opt.step()

            logging.info("\nEpoch {} loss: {}".format(m, sum(loss) / len(loss)))
            logging.info("Eval on training data:")
            self.validate(training_data_np, batch_size)
            logging.info("Eval on validation data:")
            self.validate(validation_data, batch_size)

        self.save(save_model_path)

    def _pred_fn(self, yhat):
        return torch.topk(yhat, 5, dim=-1)[1]

    def _loss_fn(self, yhat, y):
        return nn.CrossEntropyLoss(weight=self.tag_weights)(yhat.double(), y.view(-1))

    def validate(self, validation_data, batch_size):
        """
        Compute precision @top-k on the validation set.
        """
        inputs, out = zip(*validation_data)
        pred = self._predict(inputs, batch_size, False)
        out = np.reshape(np.array(out), (-1,))
        p1, pk = precision_at(pred, out)
        logging.info("p@1: {}".format(p1))
        logging.info("p@k: {}".format(pk))

    def _predict(self, cubes, batch_size, soft):
        """
        Given a list of cubes from a np_blocks, this function predicts tags by
        minibatches and then concatenates them.

        If soft=True, then only outputs floating belief numbers;
        otherwise outputs discrete tags.

        The output is torch tensor.
        """
        self.eval()

        pred = []
        for i in range(0, len(cubes), batch_size):
            batch = cubes[i : i + batch_size]
            ipt = torch.from_numpy(np.stack(batch, axis=0))
            if self.is_cuda():
                ipt = ipt.cuda(self.opts["gpu_id"])
            if soft:
                out = self(ipt)
            else:
                out = self._pred_fn(self(ipt))
            # convert from tensor to numpy to avoid CUDA memory overflow
            pred.append(out.detach().cpu().numpy())

        return np.concatenate(pred, axis=0)

    def segment_object(self, np_blocks, batch_size):
        """
        Given a numpy array of an object, the LocalVoxelCNN first predicts class
        scores at each location, then the CRF does an inference to smooth the
        result.
        This model can only do semantic segmentation; it cannot separate neighboring
        instances of the same class.
        """

        def clip_score(s, threshold=None):
            if threshold is None:
                return s
            s[s <= threshold] = threshold
            return s

        # first predict unary scores at each location
        xs, ys, zs = np.nonzero(np_blocks[:, :, :, 0] > 0)
        locs = list(zip(*[xs, ys, zs]))
        cubes = get_batch_voxel_center_cubes(locs, np_blocks, self.cube_size)
        scores = self._predict(cubes, None, batch_size, soft=True)
        scores = dict(zip(locs, scores))
        unary = {l: -clip_score(s, -10) for l, s in scores.items()}  ## minimizing the potential

        # read the embedding table and use embeddings as CRF features
        if not mbi.voxel_group:
            embed_table = self.embedding.weight.data.detach().cpu().numpy()
            feat_dim = embed_table.shape[-1]
            # reshape in case the embedding table is a 1x1 conv kernel
            embed_table = embed_table.reshape((mbi.total_ids_n(), feat_dim))
        features = {}
        for l in unary.keys():
            x, y, z = l
            if mbi.voxel_group:
                voxel_id = np_blocks[x, y, z, 0]
                features[l] = mbi.random_embedding(voxel_id)
            else:
                voxel_id = mbi.check_block_id()(np_blocks[x, y, z, 0])
                features[l] = embed_table[voxel_id]

        return self.crf.inference(unary, features)

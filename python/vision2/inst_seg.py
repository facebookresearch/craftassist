import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from voxel_cnn_base import VoxelCNN
from voxel_cube import make_center
import mc_block_ids as mbi


def compute_bce_f1(pred, gt):
    """
    Pred and gt are both binary torch.LongTensor of the same sizes.
    Return the F1 score.
    """
    assert pred.nelement() == gt.nelement()
    pred = pred.view(-1).cpu().numpy()
    gt = gt.view(-1).cpu().numpy()
    idx = np.nonzero(pred)
    t = gt[idx]

    # no positive groundtruth proposals
    # and thus no meaning for f1 in this case, just ignore
    if sum(gt) == 0:
        return

    recall = sum(t) / sum(gt)
    if t.shape[0] == 0:
        precision = 0
    else:
        precision = sum(t) / t.shape[0]
    assert recall >= 0 and recall <= 1 and precision >= 0 and precision <= 1
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return f1


def compute_acc(pred, gt):
    """
    Pred and gt are both torch.LongTensor of the same sizes.
    When computing the average accuracy, we weight each class 'g' equally,
    i.e., we first compute the average within-class accuracies and then compute
    the average across-class accuracy.
    """
    n = pred.nelement()
    assert n == gt.nelement()
    pred = pred.view(-1).cpu().numpy()
    gt = gt.view(-1).cpu().numpy()

    acc = {}
    for p, g in zip(pred, gt):
        if g not in acc:
            acc[g] = []
        acc[g].append(float(p == g))

    for g in acc.keys():
        acc[g] = np.mean(acc[g])
    return np.mean(list(acc.values()))


class InstSegModel(VoxelCNN):
    """
    This is the v1 vision model.
    Please refer to
      https://fb.quip.com/eCdkAK9cHZoC#LKdACA0iVZ1
    for an overview of the model and a better understanding of
    the code comments.
    """

    def __init__(self, tags=[], opts={}, model_path=None):
        # how many encoding/decoding layers the U-net has
        self.n_scales = opts.get("n_scales", 4)
        super().__init__(opts, tags, model_path)
        self.none_id = [t for t, n in self.tags].index("nothing")
        # [semantic_cost_w, inst_cost_w]
        self.cost_weights = opts.get("cost_weights", [0.1, 1])

    def _init_net(self):
        """
        See the two U-net architectures at
          https://fb.quip.com/eCdkAK9cHZoC#LKdACAcOi6o
        """
        voxel_embedding_dim = self.opts["voxel_embedding_dim"]

        # embedding layer for the input object cube X
        self.embedding = nn.Embedding(mbi.total_ids_n(), voxel_embedding_dim)
        # embedding layer for S that contains the seed voxel
        # with its label id
        self.cls_embedding = nn.Embedding(len(self.tags), voxel_embedding_dim)

        strides = [2] * self.n_scales
        channels = [128] * self.n_scales

        # U_sem:  self.semantic_layers + self.semantic_up_layers
        # U_inst: self.inst_layers + self.inst_up_layers
        self.semantic_layers = nn.ModuleList()
        self.inst_layers = nn.ModuleList()
        self.semantic_up_layers = nn.ModuleList()
        self.inst_up_layers = nn.ModuleList()

        def _define_encoder(layers, depth):
            ###################
            # scale     size
            # 0         16
            # 1         8
            # 2         4
            # 3         2
            ###################
            c = voxel_embedding_dim * depth
            for i in range(self.n_scales):
                module = nn.Sequential(
                    nn.Conv3d(c, channels[i], kernel_size=3, stride=strides[i], padding=1),
                    nn.BatchNorm3d(channels[i]),
                    nn.ReLU(),
                )
                c = channels[i]
                layers.append(module)

        def _define_decoder(up_layers):
            for i in range(1, self.n_scales):
                if i == self.n_scales - 1:
                    deconv_in_dim = channels[i]
                else:
                    deconv_in_dim = channels[i] * 2
                up_layers.append(
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

        _define_encoder(self.semantic_layers, depth=1)
        _define_encoder(self.inst_layers, depth=2)
        _define_decoder(self.semantic_up_layers)
        _define_decoder(self.inst_up_layers)

        # Semantic output layer
        # The output channel would be the total number of label classes
        self.semantic_out_layer = nn.ConvTranspose3d(
            channels[0] * 2,
            len(self.tags),
            kernel_size=3,
            stride=strides[0],
            padding=1,
            output_padding=1,
        )

        # Instance mask output layer
        # The output will be single-channel which represents sigmoid logits
        self.mask_out_layer = nn.ConvTranspose3d(
            channels[0] * 2, 1, kernel_size=3, stride=strides[0], padding=1, output_padding=1
        )

    def forward(self, x, mask=None):
        """
        'mask' is the 3D cube S that contains the seed voxel.
        When mask is None, we only perform semantic segmentation
        otherwise, we perform both semantic and instance segmentations.
        """

        def _encode(x, layers):
            """
            This encode function applies encoder layers in sequence
            """
            temps = []
            z = x
            for i in range(self.n_scales):
                z = layers[i](z)
                temps.append(z)
            return temps

        def _decode(temps, up_layers, out_layer):
            """
            This decode function applies decoder layers in sequence
            """
            for i in range(self.n_scales - 1, 0, -1):
                x = 0  # FIXME ADS for flake
                if i == self.n_scales - 1:
                    z = temps[-1]
                else:
                    z = torch.cat((x, temps[i]), dim=1)
                x = up_layers[i - 1](z)
            return out_layer(torch.cat((x, temps[0]), dim=1))

        B, H, W, D = x.size()
        x = self.embedding(x.long())  # B, H, W, D, C
        x = x.permute(0, 4, 1, 2, 3)  # B, C, H, W, D

        if mask is not None:
            """
            If the seed is provided, we first feed 'mask' to an embedding layer
            """
            mask = self.cls_embedding(mask)
            mask = mask.permute(0, 4, 1, 2, 3)

        # semantic encoding
        semantic_temps = _encode(x, self.semantic_layers)
        # inst encoding
        if mask is not None:
            z = torch.cat((x, mask), dim=1)
            inst_temps = _encode(z, self.inst_layers)

        # semantic decoding
        semantic_out = _decode(semantic_temps, self.semantic_up_layers, self.semantic_out_layer)

        # inst decoding
        mask_out = None
        if mask is not None:
            mask_out = _decode(inst_temps, self.inst_up_layers, self.mask_out_layer)

        return semantic_out, mask_out

    def get_non_air_mask(self, x):
        # ignore air blocks when computing prediction/loss
        return (x.unsqueeze(dim=1) != 0).float()

    def forward_and_loss(self, x, mask, semantic_target, mask_target):
        """
        Given the input 'x', the seed representation 'mask', the semantic
        segmentation groundtruth 'semantic_target', and the instance
        segmentation groundtruth 'mask_target', this function computes
        C_sem and C_inst
        """
        ## only predict at non-air blocks
        non_air_mask = self.get_non_air_mask(x).view(-1)
        idx = torch.nonzero(non_air_mask).split(1, dim=1)
        semantic_out, mask_out = self(x, mask)
        semantic_out = semantic_out.permute(0, 2, 3, 4, 1).contiguous()
        semantic_out = semantic_out.view(-1, semantic_out.size()[-1])
        semantic_loss = nn.CrossEntropyLoss(weight=self.tag_weights.float())(
            semantic_out[idx].squeeze(1), semantic_target.view(-1)[idx].squeeze(1)
        )
        mask_loss = nn.BCEWithLogitsLoss()(mask_out.view(-1)[idx], mask_target.view(-1)[idx])
        return semantic_loss, mask_loss

    def predict(self, x, mask):
        """
        Prediction on the validation set, where semantic segmentation
        groundtruth is known.
        """
        ## only predict at non-air blocks
        non_air_mask = self.get_non_air_mask(x)
        semantic_out, mask_out = self(x, mask)
        # manually set air voxel locations to 0s
        mask_out = (mask_out > 0).float() * non_air_mask
        _, semantic_out = semantic_out.max(dim=1)
        idx = torch.nonzero(1 - non_air_mask.squeeze(1)).split(1, dim=1)
        # manually set air voxel locations to 'nothing'
        semantic_out[idx] = self.none_id
        return semantic_out, mask_out.squeeze(1)

    def _test_predict(self, x):
        """
        Prediction in reality, where two stages are performed: the first
        stage sample a seed voxel with its label, and the second stage
        does instance segmentation based on the sampled input.
        """
        assert x.size()[0] == 1
        # Stage 1: semantic segmentation
        semantic_out, _ = self(x)
        _, semantic_out = semantic_out.max(dim=1)

        non_air_mask = self.get_non_air_mask(x)
        idx = torch.nonzero(1 - non_air_mask.squeeze(1)).split(1, dim=1)
        semantic_out[idx] = self.none_id

        # get all indices that are not predicted as 'nothing'
        idx = torch.nonzero(semantic_out != self.none_id)
        if idx.size()[0] == 0:  # all predicted as 'nothing'
            return

        # randomly sample a voxel location
        idx = idx[torch.randperm(idx.size()[0])]
        idx = idx[:1].split(1, dim=1)
        # set the predicted label to the voxel location
        mask = torch.zeros_like(x, dtype=torch.long)
        mask[idx] = semantic_out[idx]

        # Stage 2: instance segmentation
        _, mask_out = self(x, mask)
        mask_out = (mask_out > 0).float() * non_air_mask
        mask_out = mask_out.squeeze(0).squeeze(1)  # H, W, D
        return torch.nonzero(mask_out), int(semantic_out[idx])

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
            loss = []

            xs, masks, semantic_targets, mask_targets = self.sample_an_epoch(training_data, 8)
            rperm = torch.randperm(xs.size()[0])
            xs = xs[rperm]
            masks = masks[rperm]
            semantic_targets = semantic_targets[rperm]
            mask_targets = mask_targets[rperm]

            for i in range(0, xs.size()[0], batch_size):
                if i % batch_size == 0:
                    print(".", end="", flush=True)

                x = xs[i : i + batch_size]
                mask = masks[i : i + batch_size]
                semantic_target = semantic_targets[i : i + batch_size]
                mask_target = mask_targets[i : i + batch_size]

                if self.is_cuda():
                    x = x.cuda()
                    mask = mask.cuda()
                    semantic_target = semantic_target.cuda()
                    mask_target = mask_target.cuda()

                opt.zero_grad()
                ls = self.forward_and_loss(x, mask, semantic_target, mask_target)
                ls = [l * w for l, w in zip(ls, self.cost_weights)]
                loss.append(ls)
                l = sum(ls)
                l.backward()
                opt.step()

            logging.info("\nEpoch {}".format(m))
            semantic_loss, mask_loss = zip(*loss)
            logging.info("Semantic Loss: {}".format(sum(semantic_loss) / len(semantic_loss)))
            logging.info("Mask Loss: {}".format(sum(mask_loss) / len(mask_loss)))
            logging.info("Eval on training data:")
            self.validate((xs, masks, semantic_targets, mask_targets), batch_size)
            logging.info("Eval on validation data:")
            self.validate(self.sample_an_epoch(validation_data, 16), batch_size)

        self.save(save_model_path)

    def sample_an_epoch(self, data, samples_per_house):
        """
        Given a dataset 'data', for each sample house we sample 'samples_per_house'
        seed voxels according to the description at:
          https://fb.quip.com/eCdkAK9cHZoC#LKdACAGKKKi
        """

        def _sample_minibatch(obj, inst_anno, cls_anno, batch_size):
            idx = torch.nonzero(inst_anno > 0)
            if idx.nelement() == 0:
                return [None] * 4
            idx = idx[torch.randperm(idx.size()[0])]
            idx = idx[:batch_size]  # only sample a minibatch
            B = idx.size()[0]
            x = torch.stack([obj] * B, dim=0)
            cls_target = torch.stack([cls_anno] * B, dim=0)
            mask = torch.zeros_like(x, dtype=torch.long)
            idx = idx.split(1, dim=1)
            bidx = torch.arange(0, B).unsqueeze(-1)
            idx_ = (bidx.long(), idx[0], idx[1], idx[2])
            mask[idx_] = cls_target[idx_]  # set label ids to seed voxel locations

            target = inst_anno[idx]
            target = (
                torch.stack([inst_anno] * B, dim=0).view(B, -1)
                == target.repeat(1, inst_anno.nelement())
            ).float()
            return x, mask, cls_target, target.view(mask.size())

        x, mask, semantic_target, mask_target = [], [], [], []
        for obj, inst_anno, cls_anno in data:
            x_, mask_, semantic_target_, mask_target_ = _sample_minibatch(
                obj, inst_anno, cls_anno, samples_per_house
            )
            if x_ is None:
                continue
            x.append(x_)
            mask.append(mask_)
            semantic_target.append(semantic_target_)
            mask_target.append(mask_target_)

        return (
            torch.cat(x, dim=0),
            torch.cat(mask, dim=0),
            torch.cat(semantic_target, dim=0),
            torch.cat(mask_target, dim=0),
        )

    def validate(self, validation_data, batch_size):
        """
        For validation, we always assume that the semantic segmentation
        information is provided for instance segmentation.
        """
        self.eval()

        xs, masks, semantic_targets, mask_targets = validation_data
        f1s = []
        accs = []
        for i in range(0, xs.size()[0], batch_size):
            x = xs[i : i + batch_size]
            mask = masks[i : i + batch_size]
            semantic_target = semantic_targets[i : i + batch_size]
            mask_target = mask_targets[i : i + batch_size]

            if self.is_cuda():
                x = x.cuda()
                mask = mask.cuda()
                semantic_target = semantic_target.cuda()
                mask_target = mask_target.cuda()

            with torch.no_grad():
                semantic_out, mask_out = self.predict(x, mask)

            non_air_mask = self.get_non_air_mask(x).squeeze(1)
            for batch_idx in range(x.size()[0]):
                idx = torch.nonzero(non_air_mask[batch_idx]).split(1, dim=1)
                accs.append(
                    compute_acc(semantic_out[batch_idx][idx], semantic_target[batch_idx][idx])
                )
                f1s.append(compute_bce_f1(mask_out[batch_idx][idx], mask_target[batch_idx][idx]))

        logging.info("Semantic Acc: {}".format(np.mean(accs)))
        logging.info("Mask F1: {}".format(np.mean(f1s)))

    def segment_object(self, x, batch_size):
        """
        Given a numpy array of an object, the model first predicts the most
        likely class at each location, then it randomly samples a location and
        continues to predict the instance mask for that location given that class.

        Note: currently the returned prediction result is only a single instance.
              The entire block object might not be fully covered by the result.
              'nothing' instances will never be returned.
              In the future, NMS might be needed.
        """
        size = 32
        X, Y, Z = x.shape
        x = make_center(x, size)
        # compute offsets made by make_center
        offset_x = X // 2 - size // 2
        offset_y = Y // 2 - size // 2
        offset_z = Z // 2 - size // 2

        x = torch.from_numpy(x)
        x = x[:, :, :, 0].unsqueeze(0)  # remove meta id
        if self.is_cuda():
            x = x.cuda()

        xyzs, tag_id = self._test_predict(x)
        xyzs = xyzs.cpu().numpy()
        xyzs = {(xyz[0] + offset_x, xyz[1] + offset_y, xyz[2] + offset_z): tag_id for xyz in xyzs}
        return xyzs

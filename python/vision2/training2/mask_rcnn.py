import numpy as np
import pickle
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.append("..")
sys.path.append("../..")
import mc_block_ids as mbi
from voxel_cnn_base import VoxelCNN
from voxel_cube import make_center, nn_resize_cube
from anchor_generation import anchorify_parallel
import multiprocessing
from multiprocessing import set_start_method


def compute_bce_f1(pred, gt):
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


def safe_torch_cat(lst, dim):
    if not lst:
        return torch.tensor([])
    return torch.cat(lst, dim=dim)


def safe_torch_stack(lst, dim):
    if not lst:
        return torch.tensor([])
    return torch.stack(lst, dim=dim)


def filter_out_none(lst):
    return [x for x in lst if x is not None]


class Identity(nn.Module):
    def forward(self, x):
        return x


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 4, 1, 2, 3)


class ThreedMaskRCNN(VoxelCNN):
    def __init__(self, tags=[], opts={}, model_path=None):
        self.N = opts.get("N", None)
        self.S = opts.get("S", None)
        self.C = opts.get("C", 128)
        self.anchor_sizes_x = opts.get("sizes_x", [1])
        self.anchor_ratios_y = opts.get("ratios_y", [1])
        self.anchor_ratios_z = opts.get("ratios_z", [1])
        self.overlap_threshold = opts.get("overlap_threshold", 0.5)
        self.negative_sampling = opts.get("negative_sampling", -1)
        self.cost_weights = opts.get("cost_weights", [1, 1])
        self.mask_prediction_size = opts.get("mask_size", 4) * self.S
        super().__init__(opts, tags, model_path)

    ############################################################################
    ######################## Network init functions ############################
    ############################################################################
    def _init_net(self):
        voxel_embedding_dim = self.opts.get("voxel_embedding_dim", 16)
        voxel_voc = self.opts.get("voxel_voc", mbi.total_ids_n())
        isn_nlayers = self.opts.get("isn_nlayers", 2)
        num_tags = max(len(self.tags), 1)
        self.feature_cnn = self._generic_feature_cnn(
            self.N, voxel_voc, voxel_embedding_dim, self.C, n_blocks=3
        )
        # cross product
        self.K = len(self.anchor_sizes_x) * len(self.anchor_ratios_y) * len(self.anchor_ratios_z)
        self.rpn = self._region_proposal_net(self.S, self.C, self.K)
        self.classifier, self.predictor = self._inst_seg_net(self.C, num_tags, isn_nlayers)

    def _generic_feature_cnn(self, N, M, E, C, n_blocks):
        """
        This function returns an nn.Module that receives an input voxel cube
        NxNxN and converts it to an NxNxNxC feature map.

        Should be called in self.__init__()

        Input:
            N: the input cube size
            M: the voxel vocabulary size
            E: the embedding size
            C: the output feature map channels

        Output:
            fcnn: a CNN module
        """
        feat_cnn = []
        feat_cnn.append(nn.Embedding(M, E))
        feat_cnn.append(Permute())

        channels = [4 * E] * (n_blocks - 1) + [C]
        in_channels = E
        for i in range(n_blocks):
            layer = nn.Sequential(
                nn.Conv3d(in_channels, channels[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(channels[i]),
                nn.ReLU(),
            )
            in_channels = channels[i]
            feat_cnn.append(layer)

        return nn.Sequential(*feat_cnn)

    def _region_proposal_net(self, S, C, K):
        """
        Return a module that uses the generic feature map to predict proposals

        Should be called in self.__init__()

        Input:
            C: the generic feature map channels
            K: the number of anchors we define for each location

        Output:
            rpn
        """
        M = int(math.log2(S))
        assert 2 ** M == S, "S must be 2^k"
        N = max(M, 3)  # at least 3 layers

        rpn = []
        strides = [1] * (N - M) + [2] * M
        channels = [C] * (N - 1) + [K]
        in_channels = C
        for i in range(N):
            module = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(),
                nn.Conv3d(in_channels, channels[i], kernel_size=3, stride=strides[i], padding=1),
                nn.ReLU() if i < N - 1 else Identity(),
            )
            in_channels = channels[i]
            rpn.append(module)

        return nn.Sequential(*rpn)

    def _inst_seg_net(self, C, O, N):
        """
        return a module that does classification and mask prediction on a proposal
        feature map

        Should be called in self.__init__()

        Input:
            C: the channels of the proposal feature map
            O: the number of tags/classes
            N: the number of layers for both classifier and predictor

        Output:
            classifier: a CNN that classifies the given feature map to a tag
            predictor: a deconv net that predicts masks for all classes; it might
                       share some layers with the classifier
        """

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                hidden_dims = [C] * (N - 1) + [O]
                in_dim = C
                layers = []
                for i in range(N):
                    layers.append(
                        nn.Sequential(
                            nn.Linear(in_dim, hidden_dims[i]),
                            nn.ReLU() if i < N - 1 else Identity(),
                        )
                    )
                    in_dim = hidden_dims[i]
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                B, c, H, W, D = x.size()
                assert c == C, "The feature map has a different channel number"
                x = nn.AvgPool3d(kernel_size=(H, W, D))
                x = x.view(B, -1)
                return self.net(x)

        class Predictor(nn.Module):
            def __init__(self):
                super().__init__()
                channels = [C] * N
                in_channels = C
                self.layers = nn.ModuleList()
                self.up_layers = nn.ModuleList()

                for i in range(N):
                    self.layers.append(
                        nn.Sequential(
                            nn.Conv3d(
                                in_channels, channels[i], kernel_size=3, stride=2, padding=1
                            ),
                            nn.BatchNorm3d(channels[i]),
                            nn.ReLU(),
                        )
                    )
                    in_channels = channels[i]

                    if i < N - 1:
                        deconv_in_dim = channels[i] * 2
                    else:
                        deconv_in_dim = channels[i]
                    if i > 0:
                        deconv_out_dim = channels[i - 1]
                    else:
                        deconv_out_dim = O

                    self.up_layers.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(
                                deconv_in_dim,
                                deconv_out_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                            ),
                            nn.ReLU() if i > 0 else Identity(),
                        )
                    )

            def forward(self, x):
                B, c, H, W, D = x.size()
                assert c == C, "The feature map has a different channel number"
                assert c % (2 ** N) == 0

                temps = []
                for i in range(N):
                    x = self.layers[i](x)
                    temps.append(x)
                for i in range(N - 1, -1, -1):
                    if i < N - 1:
                        x = torch.cat((x, temps[i]), dim=1)
                    x = self.up_layers[i](x)
                return x

        return Classifier(), Predictor()

    ############################################################################
    ######################## training helper functions #########################
    ############################################################################
    def _anchorify(self, data):
        pool = multiprocessing.Pool(80)
        ret_data = pool.map(
            anchorify_parallel,
            [
                (
                    anno,
                    self.N,
                    self.S,
                    self.anchor_sizes_x,
                    self.anchor_ratios_y,
                    self.anchor_ratios_z,
                    self.overlap_threshold,
                )
                for _, anno in data
            ],
        )
        pool.close()
        pool.join()

        ret_data = [
            (
                torch.from_numpy(data[i][0]),
                [
                    [aa[0], aa[1], torch.from_numpy(aa[2]) if aa[2] is not None else None]
                    for aa in ret_data[i]
                ],
            )
            for i in range(len(ret_data))
        ]
        return ret_data

    def _batchify(self, data, batch_size):
        def get_proposal_groundtruth(anchors_positive):
            # the order of anchors is H,W,D,C
            proposal_gt = np.array(anchors_positive, dtype=np.float32)
            proposal_gt = torch.from_numpy(
                np.reshape(proposal_gt, (self.N // self.S, self.N // self.S, self.N // self.S, -1))
            )
            return proposal_gt.permute(3, 0, 1, 2)

        expand_data = []
        for obj, annotated_anchors in data:
            proposal_gt = get_proposal_groundtruth([aa[1] for aa in annotated_anchors])
            # only keep positive anchors and their masks
            anchors = [aa[0] for aa in annotated_anchors if aa[1]]
            mask_target = [
                nn_resize_cube(aa[2].float(), self.mask_prediction_size)
                for aa in annotated_anchors
                if aa[1]
            ]
            mask_target = safe_torch_stack(mask_target, dim=0)
            expand_data.append((obj, proposal_gt, anchors, mask_target))

        batch_data = []
        for i in range(0, len(expand_data), batch_size):
            batch = expand_data[i : i + batch_size]
            obj, proposal_gt, anchors_multiple, mask_target = zip(*batch)
            obj = torch.stack(obj, dim=0)
            proposal_gt = torch.stack(proposal_gt, dim=0)
            mask_target = safe_torch_cat(filter_out_none(mask_target), dim=0)
            batch_data.append((obj, proposal_gt, anchors_multiple, mask_target))

        return batch_data

    def _get_anchor_feature(self, feat_map, anchors_multiple):
        B, C, X, Y, Z = feat_map.size()
        assert B == len(anchors_multiple), "Unmatched lengths!"
        anchor_maps, anchor_features = [], []
        for i, anchors in enumerate(anchors_multiple):
            for anchor in anchors:
                mx, Mx = max(0, anchor.x), min(X - 1, anchor.tx)
                my, My = max(0, anchor.y), min(Y - 1, anchor.ty)
                mz, Mz = max(0, anchor.z), min(Z - 1, anchor.tz)
                chunk = feat_map[i, :, mx : Mx + 1, my : My + 1, mz : Mz + 1]

                pool = nn.AvgPool3d(kernel_size=(Mx - mx + 1, My - my + 1, Mz - mz + 1))
                af = pool(chunk).view(-1)

                # pad in the reverse order
                padding = nn.ConstantPad3d(
                    (
                        mz - anchor.z,
                        anchor.tz - Mz,
                        my - anchor.y,
                        anchor.ty - My,
                        mx - anchor.x,
                        anchor.tx - Mx,
                    ),
                    0,
                )
                am = padding(chunk).contiguous()
                am = nn_resize_cube(am, self.mask_prediction_size)

                anchor_features.append(af)
                anchor_maps.append(am)

        return safe_torch_stack(anchor_maps, dim=0), safe_torch_stack(anchor_features, dim=0)

    def _decode_anchors(self, idx):
        idx = idx[:, 1]  # the second column is the channel dim (the first is the batch dim)
        npos_by_scales = []
        for k in range(self.K):
            npos_by_scales.append(int(sum((k == idx).int())))
        assert sum(npos_by_scales) == idx.size()[0], "Unexpected channel"
        return npos_by_scales

    ###############################################################################
    ############################## training & validation ##########################
    ###############################################################################
    def train_epochs(
        self,
        training_data,
        validation_data,
        save_model_path,
        batch_size=32,
        lr=1e-4,
        max_epochs=100,
    ):
        def compute_proposal_loss(proposal_scores, proposal_gt):
            one_idx = torch.nonzero(proposal_gt == 1)
            zero_idx = torch.nonzero(proposal_gt == 0)
            if self.negative_sampling > 0:
                negative_samples = int(max(1, one_idx.size()[0]) * self.negative_sampling)
                zero_idx = zero_idx[torch.randperm(zero_idx.size()[0])]
                # compute which costs to mask out
                zero_idx = zero_idx[:negative_samples]

            idx = torch.cat((one_idx, zero_idx), dim=0)
            # both scores and gt will be 1d after selection
            proposal_scores = proposal_scores[idx.split(1, dim=1)]
            proposal_gt = proposal_gt[idx.split(1, dim=1)]
            return nn.BCEWithLogitsLoss()(proposal_scores.view(-1), proposal_gt.view(-1)), one_idx

        def compute_mask_loss(pred_mask, mask_target):
            assert mask_target.nelement() == pred_mask.nelement()
            return nn.BCEWithLogitsLoss()(pred_mask.view(-1), mask_target.view(-1))

        opt = optim.Adam(self.parameters(), lr=lr)

        training_data = self._batchify(self._anchorify(training_data), batch_size)
        validation_data = self._batchify(self._anchorify(validation_data), batch_size)

        for m in range(max_epochs):
            self.train()  ## because we do prediction after each epoch
            random.shuffle(training_data)

            loss = []
            n_positives = []
            for batch_idx, batch in enumerate(training_data):
                print(".", end="", flush=True)

                obj, proposal_gt, anchors_multiple, mask_target = batch
                if self.is_cuda():
                    obj = obj.cuda()
                    proposal_gt = proposal_gt.cuda()
                    mask_target = mask_target.cuda()

                feat_map = self.feature_cnn(obj)
                proposal_scores = self.rpn(feat_map)
                loss_p, pos_idx = compute_proposal_loss(proposal_scores, proposal_gt)
                n_positives.append(self._decode_anchors(pos_idx))

                am, af = self._get_anchor_feature(feat_map, anchors_multiple)
                if am.nelement() > 0:
                    pred_mask = self.predictor(am)
                    loss_m = compute_mask_loss(pred_mask, mask_target)
                else:
                    loss_m = None
                ###############################
                # TODO: also get class info from the annotation
                # loss_c = compute_class_loss(self.classifier(af))
                ###############################
                opt.zero_grad()
                ls = [loss_p, loss_m]
                # apply weights to the losses
                ls = [(None if l is None else l * w) for l, w in zip(ls, self.cost_weights)]
                loss.append(ls)
                l = sum(filter_out_none(ls))
                l.backward()
                opt.step()

            print("\nEpoch {}".format(m))
            for i, npos in enumerate(zip(*n_positives)):
                print(
                    "Scale {} average positives per batch: {}".format(
                        i + 1, sum(npos) / len(training_data)
                    )
                )

            for i, ls in enumerate(zip(*loss)):
                if i == 0:
                    name = "Proposal"
                elif i == 1:
                    name = "Mask"
                ls = filter_out_none(ls)
                print("{} loss: {}".format(name, sum(ls) / len(ls)))

            print("Eval on training data:")
            self.validate(training_data)
            print("Eval on validation data:")
            self.validate(validation_data)

        self.save(save_model_path)

    def validate(self, validation_data):
        self.eval()
        proposal_f1s, mask_f1s = [], []
        for batch in validation_data:
            obj, proposal_gt, anchors_multiple, mask_target = batch
            if self.is_cuda():
                obj = obj.cuda()
                proposal_gt = proposal_gt.cuda()
                mask_target = mask_target.cuda()

            with torch.no_grad():
                feat_map = self.feature_cnn(obj)
                proposal_scores = self.rpn(feat_map)
                for i in range(obj.size()[0]):
                    proposal_f1s.append(compute_bce_f1(proposal_scores[i] > 0, proposal_gt[i]))

                am, _ = self._get_anchor_feature(feat_map, anchors_multiple)
                if am.nelement() > 0:
                    pred_mask = self.predictor(am)
                    for i in range(pred_mask.size()[0]):
                        mask_f1s.append(compute_bce_f1(pred_mask[i] > 0, mask_target[i]))
                else:  # no positive groundtruth proposals for this batch
                    mask_f1s.append(None)

        proposal_f1s = filter_out_none(proposal_f1s)
        mask_f1s = filter_out_none(mask_f1s)
        print("Proposal prediction f1: {}".format(np.mean(proposal_f1s)))
        print("Mask prediction f1: {}".format(np.mean(mask_f1s)))


if __name__ == "__main__":
    set_start_method("spawn", force=True)

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

    model = ThreedMaskRCNN(
        opts=dict(
            voxel_embedding_dim=16,
            N=size,
            C=64,
            S=2,
            mask_size=8,
            sizes_x=[2, 4, 8],
            ratios_y=[1],
            ratios_z=[1],
            overlap_threshold=0.7,
            voxel_voc=200,
            cost_weights=[1, 1],
            rpn_nlayers=3,
            isn_nlayers=3,
            gpu_id=0,
        )
    )
    model.train_epochs(training_data, validation_data, "/tmp/model.pth", batch_size=16, lr=1e-3)

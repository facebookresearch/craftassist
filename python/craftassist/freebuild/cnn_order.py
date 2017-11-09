import torch
from torch import nn
import numpy as np

import utils


class ConvBlock(nn.Module):
    """
    Convolution Block contains: conv, batch-norm and relu
    """

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


class CNN(nn.Module):
    """
    Model of predicting next (x, y, z) building action based on all previous actions
    """

    def __init__(self, args, block=ConvBlock):
        """
        block: basic conv building block
        dim: shape of the input block, dim = 7 means the 3D-block is 7 x 7 x 7
        """
        super(CNN, self).__init__()
        self.multi_res = args.multi_res
        self.ldim = args.local_bsize
        self.gdim1 = args.global_bsize
        self.gdim2 = args.global_bsize // 3

        # NB(demiguo): let's make this assumption for FCN
        assert self.gdim2 == self.ldim

        self.num_block_type = args.block_id
        self.global_num_block_type = args.global_block_id
        self.f_dim = args.f_dim
        self.history = args.history
        self.embed = args.embed
        self.loss_type = args.loss_type
        # TODO(demiguo): add them in main file as arguments
        self.emb_size = args.emb_size

        # block embedding
        self.block_embeds = nn.Embedding(self.num_block_type, self.emb_size)

        ## local encoder
        self.local_in_dim = self.num_block_type * self.history if not self.embed else self.history
        if self.embed:
            """
            self.local_embed_conv = nn.Conv3d(self.local_in_dim, 3, 1, bias=False)
            self.local_conv = block(3, self.f_dim, 3)
            """
            self.local_conv = block(self.local_in_dim * self.emb_size, self.f_dim, 3)
        else:
            self.local_conv = block(self.local_in_dim, self.f_dim, 3)
        # init conv
        self.relu = nn.ReLU()
        # 3 conv layer
        self.local_layer1 = block(self.f_dim, self.f_dim, 3)
        self.local_layer2 = block(self.f_dim, self.f_dim, 3)
        self.local_layer3 = block(self.f_dim, self.f_dim, 3)
        # fc
        # self.local_out_dim = self.ldim ** 3 * self.f_dim
        # self.local_fc = nn.Linear(self.local_out_dim, self.f_dim * 2)

        ## global encoder
        self.global_in_dim = self.global_num_block_type if not self.embed else 1
        if self.embed:
            self.global_conv = block(self.global_in_dim * self.emb_size, self.f_dim, 3)
        else:
            self.global_conv = block(self.global_num_block_type, self.f_dim, 3)
        self.global_layer1 = block(self.f_dim, self.f_dim, 3)
        self.global_layer2 = nn.MaxPool3d(3)
        self.global_layer3 = block(self.f_dim, self.f_dim, 3)
        # self.global_out_dim = self.gdim2 ** 3 * self.f_dim
        # self.global_fc = nn.Linear(self.global_out_dim, self.f_dim * 2)

        ## combined
        # self.concat_fc = nn.Linear(self.f_dim * 4, self.f_dim * 2)
        self.concat_fc = nn.Linear(self.f_dim * 2, self.f_dim)

        # output
        # self.fc_coord = nn.Linear(self.f_dim * 2, self.ldim ** 3)
        # self.fc_block_id = nn.Linear(self.f_dim * 2, self.num_block_type)
        self.fc_coord = nn.Conv3d(self.f_dim, 1, 1, stride=1, bias=True)
        self.fc_block_id = nn.Linear(self.ldim ** 3 * self.f_dim, self.num_block_type)
        if self.loss_type == "conditioned":
            self.fc_block_id_cond = nn.Linear(self.f_dim, self.num_block_type)
        # self.fc_block_emb = nn.Linear(self.f_dim * 2, self.ldim ** 3 * self.emb_size)
        self.fc_block_emb = nn.Linear(self.f_dim, self.emb_size)

    def load_ckpt(self, ckpt_path):
        map_location = "cpu" if not torch.cuda.is_available() else None
        self.load_state_dict(torch.load(ckpt_path, map_location=map_location))

    def train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def encode_local(self, x):
        if self.embed:
            x = x.long()
            assert x.size() == (
                self.batch_size,
                self.local_in_dim,
                self.ldim,
                self.ldim,
                self.ldim,
            )
            out = self.block_embeds(x)
            assert out.size() == (
                self.batch_size,
                self.local_in_dim,
                self.ldim,
                self.ldim,
                self.ldim,
                self.emb_size,
            )
            out = out.permute(5, 0, 1, 2, 3, 4).transpose(0, 1).transpose(1, 2)
            assert out.size() == (
                self.batch_size,
                self.local_in_dim,
                self.emb_size,
                self.ldim,
                self.ldim,
                self.ldim,
            )
            out = out.contiguous().view(
                self.batch_size, self.local_in_dim * self.emb_size, self.ldim, self.ldim, self.ldim
            )
        else:
            out = x
        out = self.local_conv(out)
        out = self.local_layer1(out)
        out = self.local_layer2(out)
        out = self.local_layer3(out)
        return out

    def encode_global(self, y):
        if self.embed:
            y = y.long()
            assert y.size() == (
                self.batch_size,
                self.global_in_dim,
                self.gdim1,
                self.gdim1,
                self.gdim1,
            )
            out = self.block_embeds(y)
            assert out.size() == (
                self.batch_size,
                self.global_in_dim,
                self.gdim1,
                self.gdim1,
                self.gdim1,
                self.emb_size,
            )
            out = out.permute(5, 0, 1, 2, 3, 4).transpose(0, 1).transpose(1, 2)
            assert out.size() == (
                self.batch_size,
                self.global_in_dim,
                self.emb_size,
                self.gdim1,
                self.gdim1,
                self.gdim1,
            )
            out = out.contiguous().view(
                self.batch_size,
                self.global_in_dim * self.emb_size,
                self.gdim1,
                self.gdim1,
                self.gdim1,
            )
        else:
            out = y
        out = self.global_conv(out)
        assert out.size() == (self.batch_size, self.f_dim, self.gdim1, self.gdim1, self.gdim1)
        out = self.global_layer1(out)
        assert out.size() == (self.batch_size, self.f_dim, self.gdim1, self.gdim1, self.gdim1), (
            " out.size()=%s" % out.size()
        )
        out = self.global_layer2(out)
        assert out.size() == (self.batch_size, self.f_dim, self.gdim2, self.gdim2, self.gdim2), (
            " out.size()=%s" % out.size()
        )
        out = self.global_layer3(out)
        assert out.size() == (self.batch_size, self.f_dim, self.gdim2, self.gdim2, self.gdim2)
        return out

    def forward(self, x, y):
        # TODO: try not to use extra FC layers
        self.batch_size = x.size(0)
        out_local = self.encode_local(x)
        if self.multi_res:
            out_global = self.encode_global(y)
            out = torch.cat([out_local, out_global], 1)
            assert self.ldim == self.gdim2  # redundant sanity check; reminder
            assert out.size() == (self.batch_size, self.f_dim * 2, self.ldim, self.ldim, self.ldim)
            out = out.permute(1, 2, 3, 4, 0).transpose(0, 4)
            assert out.size() == (self.batch_size, self.ldim, self.ldim, self.ldim, self.f_dim * 2)
            out = self.concat_fc(out)
            assert out.size() == (self.batch_size, self.ldim, self.ldim, self.ldim, self.f_dim)
            out = out.permute(4, 0, 1, 2, 3).transpose(0, 1)
            assert out.size() == (self.batch_size, self.f_dim, self.ldim, self.ldim, self.ldim)
        else:
            out = out_local

        out_coord = self.fc_coord(out)
        assert out_coord.size() == (self.batch_size, 1, self.ldim, self.ldim, self.ldim)
        out_coord = out_coord.view(self.batch_size, self.ldim ** 3)
        if self.loss_type == "regular":
            out_block_id = out.contiguous().view(self.batch_size, self.f_dim * self.ldim ** 3)
            out_block_id = self.fc_block_id(out_block_id)
            assert out_block_id.size() == (self.batch_size, self.num_block_type)
        elif self.loss_type == "regression":
            out_block_id = out.permute(1, 2, 3, 4, 0).transpose(0, 4)
            assert out_block_id.size() == (
                self.batch_size,
                self.ldim,
                self.ldim,
                self.ldim,
                self.f_dim,
            )
            out_block_id = out_block_id.contiguous().view(
                self.batch_size * self.ldim ** 3, self.f_dim
            )
            out_block_id = self.fc_block_emb(out_block_id)
            assert out_block_id.size() == (self.batch_size * self.ldim ** 3, self.emb_size)
            out_block_id = out_block_id.contiguous().view(
                self.batch_size, self.ldim ** 3, self.emb_size
            )
        elif self.loss_type == "conditioned":
            out_block_id = out.permute(1, 2, 3, 4, 0).transpose(0, 4)
            assert out_block_id.size() == (
                self.batch_size,
                self.ldim,
                self.ldim,
                self.ldim,
                self.f_dim,
            )
            out_block_id = out_block_id.view(self.batch_size, self.ldim ** 3, self.f_dim)
        return out_coord, out_block_id

    def get_reg_loss(self, out1, out2, target_next_steps):
        # convert targets into desirable formats
        target_next_steps_coord, target_next_steps_type = target_next_steps
        mtarget1 = torch.zeros((self.batch_size, self.ldim ** 3), dtype=torch.long)
        mtarget2 = np.zeros((self.batch_size, self.ldim ** 3), dtype=np.int64)
        # NB: we tentatively use brute force
        for b in range(self.batch_size):
            for i in range(len(target_next_steps_coord[b])):
                if i >= self.train_next_steps:
                    break
                assert (
                    target_next_steps_coord[b][i] < self.ldim ** 3
                ), " target next[%d][%d]=%s" % (b, i, target_next_steps_coord[b][i])
                mtarget1[b][target_next_steps_coord[b][i]] = 1
                mtarget2[b][target_next_steps_coord[b][i]] = target_next_steps_type[b][i]
        mtarget2 = torch.from_numpy(mtarget2)
        if torch.cuda.is_available():
            mtarget1 = mtarget1.cuda()
            mtarget2 = mtarget2.cuda()
        # TODO: for losses, do we need averaging here?
        # loss for coordinates
        out1_flat = out1.view(self.batch_size * self.ldim ** 3, 1)
        target1_flat = mtarget1.view(self.batch_size * self.ldim ** 3, 1).float()
        loss1 = nn.BCEWithLogitsLoss()(out1_flat, target1_flat)

        # loss for block types
        mtarget_emb = self.block_embeds(mtarget2).view(
            self.batch_size, self.ldim ** 3, self.emb_size
        )
        assert out2.size() == (self.batch_size, self.ldim ** 3, self.emb_size)
        dist = (mtarget_emb - out2) ** 2
        l2norm = torch.sqrt(torch.sum(dist, dim=2) + 1e-8)
        assert l2norm.size() == (self.batch_size, self.ldim ** 3)
        masked_l2norm = l2norm * mtarget1.view(self.batch_size, self.ldim ** 3).float()
        loss2 = torch.sum(masked_l2norm) / torch.sum(mtarget1.float())

        loss = loss1 + self.reg_loss_scale * loss2
        # print("loss1=", loss1, " loss2=", loss2)
        return loss

    def loss(self, out1, out2, target1, target2, target_next_steps, train=True):
        if self.loss_type == "regular":
            if self.num_block_type > 1:
                out = [out1, out2]
                target = [target1, target2]
            else:
                out = [out1]
                target = [target1]
            loss = self.criterion(out[0], target[0])
            for i in range(1, len(out)):
                loss += self.criterion(out[i], target[i])
        elif self.loss_type == "regression":
            assert self.num_block_type == 256
            loss = self.get_reg_loss(out1, out2, target_next_steps)
        elif self.loss_type == "conditioned":
            assert self.num_block_type == 256
            loss1 = self.criterion(out1, target1)
            cond_out2 = []
            for b in range(self.batch_size):
                cond_out2.append(out2[b, target1[b], :])
            cond_out2 = torch.stack(cond_out2, dim=0)
            assert cond_out2.size() == (self.batch_size, self.f_dim)
            cond_out2 = self.fc_block_id_cond(cond_out2)
            assert cond_out2.size() == (self.batch_size, self.num_block_type)
            loss2 = self.criterion(cond_out2, target2)
            loss = loss1 + loss2
        else:
            raise NotImplementedError
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def get_reg_predict(self, out1, out2):
        prob1 = torch.nn.Sigmoid()(out1)
        assert prob1.size() == (self.batch_size, self.ldim ** 3)
        pred1 = torch.argmax(prob1, dim=1)
        assert pred1.size() == (self.batch_size,)

        assert out2.size() == (self.batch_size, self.ldim ** 3, self.emb_size)
        single_out2 = []
        for b in range(self.batch_size):
            single_out2.append(out2[b][pred1[b]])
        single_out2 = torch.stack(single_out2)
        assert single_out2.size() == (self.batch_size, self.emb_size)
        single_out2 = single_out2.view(self.batch_size, 1, self.emb_size)
        single_out2_expand = single_out2.repeat([1, self.num_block_type, 1])
        all_block_embeds = self.block_embeds.weight
        assert all_block_embeds.size() == (self.num_block_type, self.emb_size)
        all_block_embeds_expand = all_block_embeds.view(1, self.num_block_type, self.emb_size)
        all_block_embeds_expand = all_block_embeds_expand.repeat([self.batch_size, 1, 1])
        blocks_match = (single_out2_expand - all_block_embeds_expand) ** 2
        blocks_match = torch.sum(blocks_match, dim=2)
        assert blocks_match.size() == (self.batch_size, self.num_block_type)
        pred2 = torch.argmax(blocks_match, dim=1)
        assert pred2.size() == (self.batch_size,)
        return (pred1, pred2)

    def predict(self, out1, out2):
        if self.loss_type == "regular":
            return (torch.argmax(out1, dim=1), torch.argmax(out2, dim=1))
        elif self.loss_type == "regression":
            return self.get_reg_predict(out1, out2)
        elif self.loss_type == "conditioned":
            pred1 = torch.argmax(out1, dim=1)  # [batch_size, ]
            assert pred1.size() == (self.batch_size,)
            cond_out2 = []
            for b in range(self.batch_size):
                cond_out2.append(out2[b, pred1[b], :])
            cond_out2 = torch.stack(cond_out2, dim=0)
            assert cond_out2.size() == (self.batch_size, self.f_dim)
            cond_out2 = self.fc_block_id_cond(cond_out2)
            assert cond_out2.size() == (self.batch_size, self.num_block_type)
            pred2 = torch.argmax(cond_out2, dim=1)
            return (pred1, pred2)
        else:
            raise NotImplemented

    def full_predict(self, data_list):
        # make a full prediction of one sample
        # assume the running is always in eval() mode!
        # otherwise, moving mean/variance will be wrong
        local_data = utils.get_data(
            data_list,
            len(data_list) - 1,
            self.ldim,
            self.num_block_type,
            self.history,
            embed=self.embed,
        )
        local_data = local_data.reshape(
            (1, self.num_block_type * self.history, self.ldim, self.ldim, self.ldim)
        )
        if self.multi_res:
            global_in_dim = self.global_in_dim if not self.embed else 1
            global_data = utils.get_data(
                data_list, len(data_list) - 1, self.gdim1, global_in_dim, 1, embed=self.embed
            )
            global_data = global_data.reshape(
                (1, global_in_dim, self.gdim1, self.gdim1, self.gdim1)
            )
        else:
            global_data = None

        # step 2: prepare in a pytorch tensor
        local_data = torch.from_numpy(local_data)
        if self.multi_res:
            global_data = torch.from_numpy(global_data)
        if torch.cuda.is_available():
            local_data = local_data.cuda()
            if self.multi_res:
                global_data = global_data.cuda()

        # step 3: voxel-cnn, run the model
        out1, out2 = self.forward(local_data, global_data)
        pred = self.predict(out1, out2)
        # out1 back to (x, y, z)
        pred_xyz = int(pred[0])
        xp, yp, zp = (
            int(pred_xyz / self.ldim ** 2),
            int(pred_xyz / self.ldim) % self.ldim,
            pred_xyz % self.ldim,
        )
        pred_block = int(pred[1])

        xc, yc, zc = data_list[-1][0]

        local_hsize = (self.ldim - 1) // 2
        xp += xc - local_hsize
        yp += yc - local_hsize
        zp += zc - local_hsize

        # same block_id as last block, then same meta-id
        if pred_block == data_list[-1][1][0]:
            meta_id = data_list[-1][1][1]
        else:
            meta_id = 0
        return [xp, yp, zp], (pred_block, meta_id)

    def full_predict_mc(self, agent, xyz):
        # get the local context from agent
        ox, oy, oz = xyz
        h_ldim = (self.ldim - 1) // 2
        local_npy = agent.get_blocks(
            ox - h_ldim, ox + h_ldim, oy - h_ldim, oy + h_ldim, oz - h_ldim, oz + h_ldim
        )
        local_npy = np.transpose(local_npy, (2, 0, 1, 3))
        local_data = utils.npy_sparse2full(local_npy, self.num_block_type, self.history)

        if self.multi_res:
            h_gdim = (self.gdim1 - 1) // 2
            global_in_dim = self.global_in_dim if not self.embed else 1
            global_npy = agent.get_blocks(
                ox - h_gdim, ox + h_gdim, oy - h_gdim, oy + h_gdim, oz - h_gdim, oz + h_gdim
            )
            global_npy = np.transpose(global_npy, (2, 0, 1, 3))
            global_data = utils.npy_sparse2full(global_npy, 1, 1)

            global_data = global_data.reshape(
                (1, global_in_dim, self.gdim1, self.gdim1, self.gdim1)
            )
        else:
            global_data = None

        # step 2: prepare in a pytorch tensor
        local_data = torch.from_numpy(local_data)
        if self.multi_res:
            global_data = torch.from_numpy(global_data)
        if torch.cuda.is_available():
            local_data = local_data.cuda()
            if self.multi_res:
                global_data = global_data.cuda()

        # step 3: voxel-cnn, run the model
        out1, out2 = self.forward(local_data, global_data)
        pred = self.predict(out1, out2)
        # out1 back to (x, y, z)
        pred_xyz = int(pred[0])
        xp, yp, zp = (
            int(pred_xyz / self.ldim ** 2),
            int(pred_xyz / self.ldim) % self.ldim,
            pred_xyz % self.ldim,
        )
        pred_block = int(pred[1])

        xc, yc, zc = xyz

        local_hsize = (self.ldim - 1) // 2
        xp += xc - local_hsize
        yp += yc - local_hsize
        zp += zc - local_hsize

        # same block_id as last block, then same meta-id
        last_block = agent.get_blocks(ox, ox, oy, oy, oz, oz).flatten()
        if pred_block == last_block[0]:
            meta_id = last_block[1]
        else:
            meta_id = 0
        return [xp, yp, zp], (pred_block, meta_id)


if __name__ == "__main__":
    model = CNN(ConvBlock)
    model = model.cuda()
    data = torch.randn([1, 256, 7, 7, 7]).cuda()
    out1, out2 = model(data)

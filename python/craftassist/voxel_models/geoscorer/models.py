"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
import random
import string
from shutil import copyfile
from geoscorer_util import get_xyz_viewer_look_coords_batched


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


# Return an 32 x 32 x 32 x 3 tensor where each len 3 inner tensor is
# the xyz coordinates of that position
def create_xyz_tensor(sl):
    incr_t = torch.tensor(range(sl), dtype=torch.float64)
    z = incr_t.expand(sl, sl, sl).unsqueeze(3)
    y = incr_t.unsqueeze(1).expand(sl, sl, sl).unsqueeze(3)
    x = incr_t.unsqueeze(1).unsqueeze(2).expand(sl, sl, sl).unsqueeze(3)
    xyz = torch.cat([x, y, z], 3)
    return xyz


class ValueNet(nn.Module):
    def __init__(self, opts):
        super(ValueNet, self).__init__()

        self.embedding_dim = opts.get("blockid_embedding_dim", 8)
        self.num_layers = opts.get("num_layers", 4)  # 32x32x32 input
        num_words = opts.get("num_words", 3)
        hidden_dim = opts.get("hidden_dim", 64)

        self.embedding = nn.Embedding(num_words, self.embedding_dim)
        self.layers = nn.ModuleList()
        indim = self.embedding_dim
        outdim = hidden_dim
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(indim, outdim, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm3d(outdim),
                nn.ReLU(inplace=True),
            )
        )
        indim = outdim
        for i in range(self.num_layers - 1):
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


class ContextEmbeddingNet(nn.Module):
    def __init__(self, opts, blockid_embedding):
        super(ContextEmbeddingNet, self).__init__()

        self.blockid_embedding_dim = opts.get("blockid_embedding_dim", 8)
        output_embedding_dim = opts.get("output_embedding_dim", 8)
        num_layers = opts.get("num_layers", 4)
        hidden_dim = opts.get("hidden_dim", 64)
        self.use_direction = opts.get("cont_use_direction", False)
        self.use_xyz_from_viewer_look = opts.get("cont_use_xyz_from_viewer_look", False)
        self.context_sl = opts.get("context_side_length", 32)
        self.xyz = None

        input_dim = self.blockid_embedding_dim
        if self.use_direction:
            input_dim += 5
        if self.use_xyz_from_viewer_look:
            input_dim += 3
            self.xyz = create_xyz_tensor(self.context_sl).view(1, -1, 3)
            if opts.get("cuda", 0):
                self.xyz = self.xyz.cuda()

        # A shared embedding for the block id types
        self.blockid_embedding = blockid_embedding

        # Create model for converting the context into HxWxL D dim representations
        self.layers = nn.ModuleList()
        # B dim block id -> hidden dim, maintain input size
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        # hidden dim -> hidden dim, maintain input size
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        # hidden dim -> spatial embedding dim, maintain input size
        self.out = nn.Linear(hidden_dim, output_embedding_dim)

    # Input: [context, opt:viewer_pos, opt:viewer_look, opt:direction]
    # Returns N x D x H x W x L
    def forward(self, inp):
        if inp[0].size()[1] != 32 or inp[0].size()[2] != 32 or inp[0].size()[3] != 32:
            raise Exception("Size of input should be Nx32x32x32 but it is {}".format(inp.size()))

        x = inp[0]
        sizes = list(x.size())
        x = x.view(-1)
        # Get the blockid embedding for each space in the context input
        z = self.blockid_embedding.weight.index_select(0, x)
        z = z.float()
        # Add the embedding dim B
        sizes.append(self.blockid_embedding_dim)

        # z: N*D x B
        if self.use_xyz_from_viewer_look:
            viewer_pos = inp[1]
            viewer_look = inp[2]
            n = viewer_pos.size()[0]
            n_xyz = self.xyz.expand(n, -1, -1)
            # Input: viewer pos, viewer look (N x 3), n_xyz (N x D x 3)
            n_xyz = (
                get_xyz_viewer_look_coords_batched(viewer_pos, viewer_look, n_xyz)
                .view(-1, 3)
                .float()
            )
            z = torch.cat([z, n_xyz], 1)
            # Add the xyz_look_position to the input size list
            sizes[-1] += 3

        if self.use_direction:
            # direction: N x 5
            direction = inp[3]
            d = self.context_sl * self.context_sl * self.context_sl
            direction = direction.unsqueeze(1).expand(-1, d, -1).contiguous().view(-1, 5)
            direction = direction.float()
            z = torch.cat([z, direction], 1)
            # Add the direction emb to the input size list
            sizes[-1] += 5

        z = z.view(torch.Size(sizes))
        # N x H x W x L x B ==> N x B x H x W x L
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        return self.out(z)


class SegmentEmbeddingNet(nn.Module):
    def __init__(self, opts, blockid_embedding):
        super(SegmentEmbeddingNet, self).__init__()

        self.blockid_embedding_dim = opts.get("blockid_embedding_dim", 8)
        spatial_embedding_dim = opts.get("spatial_embedding_dim", 8)
        hidden_dim = opts.get("hidden_dim", 64)

        # A shared embedding for the block id types
        self.blockid_embedding = blockid_embedding

        # Create model for converting the segment into 1 D dim representation
        # input size: 8x8x8
        self.layers = nn.ModuleList()
        # B dim block id -> hidden dim, maintain input size
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(self.blockid_embedding_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        # hidden dim -> hidden dim
        #   (maintain input size x2, max pool to half) x 3: 8x8x8 ==> 1x1x1
        for i in range(3):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2, stride=2),
                )
            )
        # hidden dim -> spatial embedding dim, 1x1x1
        self.out = nn.Linear(hidden_dim, spatial_embedding_dim)

    # Returns N x D x 1 x 1 x 1
    def forward(self, x):
        if x.size()[1] != 8:
            raise Exception("Size of input should be Nx8x8x8 but it is {}".format(x.size()))
        sizes = list(x.size())
        x = x.view(-1)
        # Get the blockid embedding for each space in the context input
        z = self.blockid_embedding.weight.index_select(0, x)
        # Add the embedding dim B
        sizes.append(self.blockid_embedding_dim)
        z = z.view(torch.Size(sizes))
        # N x H x W x L x B ==> N x B x H x W x L
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        return self.out(z)


class SegmentDirectionEmbeddingNet(nn.Module):
    def __init__(self, opts):
        super(SegmentDirectionEmbeddingNet, self).__init__()

        output_embedding_dim = opts.get("output_embedding_dim", 8)
        self.use_viewer_pos = opts.get("seg_use_viewer_pos", False)
        self.use_viewer_look = opts.get("seg_use_viewer_look", False)
        self.use_direction = opts.get("seg_use_direction", False)
        hidden_dim = opts.get("hidden_dim", 64)
        num_layers = opts.get("num_seg_dir_layers", 3)
        self.seg_input_dim = opts.get("spatial_embedding_dim", 8)
        self.context_side_length = opts.get("context_side_length", 32)
        input_dim = self.seg_input_dim
        if self.use_viewer_pos:
            input_dim += 3
        if self.use_viewer_look:
            input_dim += 3
        if self.use_direction:
            input_dim += 5

        # Create model for converting the segment, viewer info,
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layers - 1):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.out = nn.Linear(hidden_dim, output_embedding_dim)

    # In: [seg_embedding, viewer_pos, viewer_look, direction]
    # Out: N x D x 1 x 1 x 1
    def forward(self, x):
        if len(x) != 4:
            raise Exception("There should be 5 elements in the input")
        if x[0].size()[1] != self.seg_input_dim:
            raise Exception("The seg spatial embed is wrong size: {}".format(x[0].size()))

        inp = [x[0]]
        normalizing_const = self.context_side_length * 1.0 / 2.0
        if self.use_viewer_pos:
            inp.append(x[1].float().div_(normalizing_const))
        if self.use_viewer_look:
            inp.append(x[2].float().div_(normalizing_const))
        if self.use_direction:
            inp.append(x[3].float())

        z = torch.cat(inp, 1)
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return self.out(z).unsqueeze(2).unsqueeze(3).unsqueeze(4)


class ContextSegmentScoringModule(nn.Module):
    def __init__(self):
        super(ContextSegmentScoringModule, self).__init__()

    def forward(self, x):
        context_emb = x[0]  # N x 32 x 32 x 32 x D
        seg_emb = x[1]  # N x 1 x 1 x 1 x D

        c_szs = context_emb.size()  # N x 32 x 32 x 32 x D
        batch_dim = c_szs[0]
        emb_dim = c_szs[4]
        num_scores = c_szs[1] * c_szs[2] * c_szs[3]

        # Prepare context for the dot product
        context_emb = context_emb.view(-1, emb_dim, 1)  # N*32^3 x D x 1

        # Prepare segment for the dot product
        seg_emb = seg_emb.view(batch_dim, 1, -1)  # N x 1 x D
        seg_emb = seg_emb.expand(-1, num_scores, -1).contiguous()  # N x 32^3 x D
        seg_emb = seg_emb.view(-1, 1, emb_dim)  # N*32^3 x 1 x D

        # Dot product & reshape
        # (K x 1 x D) bmm (K x D x 1) = (K x 1 x 1)
        out = torch.bmm(seg_emb, context_emb)
        return out.view(batch_dim, -1)


class spatial_emb_loss(nn.Module):
    def __init__(self):
        super(spatial_emb_loss, self).__init__()
        self.lsm = nn.LogSoftmax()
        self.crit = nn.NLLLoss()

    # format [scores (Nx32^3), targets (N)]
    def forward(self, inp):
        assert len(inp) == 2
        scores = inp[0]
        targets = inp[1]
        logsuminp = self.lsm(scores)
        return self.crit(logsuminp, targets)


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


def prepare_variables(b, opts):
    X = b.long()
    if opts["cuda"]:
        X = X.cuda()
    return X


def save_checkpoint(tms, metadata, opts, path):
    model_dict = {"context_net": tms["context_net"], "seg_net": tms["seg_net"]}
    if opts.get("seg_direction_net", False):
        model_dict["seg_direction_net"] = tms["seg_direction_net"]

    # Add all models to dicts and move state to cpu
    state_dicts = {}
    for model_name, model in model_dict.items():
        state_dicts[model_name] = model.state_dict()
        for n, s in state_dicts[model_name].items():
            state_dicts[model_name][n] = s.cpu()

    # Save to path
    torch.save(
        {
            "metadata": metadata,
            "model_state_dicts": state_dicts,
            "optimizer_state_dict": tms["optimizer"].state_dict(),
            "options": opts,
        },
        path,
    )


def create_context_segment_modules(opts):
    possible_params = ["context_net", "seg_net", "seg_direction_net"]

    # Add all of the modules
    emb_dict = torch.nn.Embedding(opts["num_words"], opts["blockid_embedding_dim"])
    tms = {
        "context_net": ContextEmbeddingNet(opts, emb_dict),
        "seg_net": SegmentEmbeddingNet(opts, emb_dict),
        "score_module": ContextSegmentScoringModule(),
        "lfn": spatial_emb_loss(),
    }
    if opts.get("seg_direction_net", False):
        tms["seg_direction_net"] = SegmentDirectionEmbeddingNet(opts)

    # Move everything to the right device
    if "cuda" in opts and opts["cuda"]:
        emb_dict.cuda()
        for n in possible_params:
            if n in tms:
                tms[n].cuda()

    # Setup the optimizer
    all_params = []
    for n in possible_params:
        if n in tms:
            all_params.extend(list(tms[n].parameters()))
    tms["optimizer"] = get_optim(all_params, opts)
    return tms


def load_context_segment_checkpoint(checkpoint_path, opts, backup=True, verbose=False):
    if not os.path.isfile(checkpoint_path):
        check_and_print_opts(opts, None)
        return {}

    if backup:
        random_uid = "".join(
            [random.choice(string.ascii_letters + string.digits) for n in range(4)]
        )
        backup_path = checkpoint_path + ".backup_" + random_uid
        copyfile(checkpoint_path, backup_path)
        print(">> Backing up checkpoint before loading and overwriting:")
        print("        {}\n".format(backup_path))

    checkpoint = torch.load(checkpoint_path)

    if verbose:
        print(">> Loading model from checkpoint {}".format(checkpoint_path))
        for opt, val in checkpoint["metadata"].items():
            print("    - {:>20}: {:<30}".format(opt, val))
        print("")
        check_and_print_opts(opts, checkpoint["options"])

    checkpoint_opts_dict = checkpoint["options"]
    if type(checkpoint_opts_dict) is not dict:
        checkpoint_opts_dict = vars(checkpoint_opts_dict)

    for opt, val in checkpoint_opts_dict.items():
        opts[opt] = val
    print(opts)

    trainer_modules = create_context_segment_modules(opts)
    trainer_modules["context_net"].load_state_dict(checkpoint["model_state_dicts"]["context_net"])
    trainer_modules["seg_net"].load_state_dict(checkpoint["model_state_dicts"]["seg_net"])
    trainer_modules["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
    if opts.get("seg_direction_net", False):
        trainer_modules["seg_direction_net"].load_state_dict(
            checkpoint["model_state_dicts"]["seg_direction_net"]
        )
    return trainer_modules


def get_context_segment_trainer_modules(opts, checkpoint_path=None, backup=False, verbose=False):
    trainer_modules = load_context_segment_checkpoint(checkpoint_path, opts, backup, verbose)

    if len(trainer_modules) == 0:
        trainer_modules = create_context_segment_modules(opts)
    return trainer_modules


def check_and_print_opts(curr_opts, old_opts):
    mismatches = []
    print(">> Options:")
    for opt, val in curr_opts.items():
        if opt and val:
            print("   - {:>20}: {:<30}".format(opt, val))
        else:
            print("   - {}: {}".format(opt, val))

        if old_opts and opt in old_opts and old_opts[opt] != val:
            mismatches.append((opt, val, old_opts[opt]))
            print("")

    if len(mismatches) > 0:
        print(">> Mismatching options:")
        for m in mismatches:
            print("   - {:>20}: new '{:<10}' != old '{:<10}'".format(m[0], m[1], m[2]))
            print("")
    return True if len(mismatches) > 0 else False


def get_optim(model_params, opts):
    optim_type = opts.get("optim", "adagrad")
    lr = opts.get("lr", 0.1)
    momentum = opts.get("momentum", 0.0)
    betas = (0.9, 0.999)

    if optim_type == "adagrad":
        return optim.Adagrad(model_params, lr=lr)
    elif optim_type == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=momentum)
    elif optim_type == "adam":
        return optim.Adam(model_params, lr=lr, betas=betas)
    else:
        raise Exception("Undefined optim type {}".format(optim_type))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_words", type=int, default=3, help="number of words in embedding")
    parser.add_argument("--imsize", type=int, default=32, help="imsize, use 32 or 64")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--hsize", type=int, default=64, help="hidden dim")
    opts = vars(parser.parse_args())

    net = ValueNet(opts)
    x = torch.LongTensor(7, 32, 32, 32).zero_()
    y = net(x)

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision as tv
import threed_cnn_model as model

import generate_3d_shape_data as sdata
import visdom


def build_block_eater(net, to_tags, threshold, opts):
    def block_eater(blocks, strict=[3]):
        net.eval()
        N = len(blocks)
        V = Variable(torch.zeros(N, 32, 32, 32).long())
        offsets = []
        toplevel_out = []
        for i in range(N):
            v, _, m = sdata.densify(blocks[i], (32, 32, 32), None, None, None)  # useid
            offsets.append(m)
            V[i] = torch.from_numpy(v)
            toplevel_out.append({})

        yhat = net(V)
        loc_out = {}
        for scale in range(4):
            idx = torch.nonzero(yhat[scale].data.cpu() > threshold)
            if len(idx.size()) == 0:
                n = 0
            else:
                n = idx.size(0)
            f = 2 ** (scale + 1)
            for i in range(n):
                for j in range(f):
                    for k in range(f):
                        for l in range(f):
                            oi = idx[i][0]
                            x = (f * idx[i][2] + j).item()
                            y = (f * idx[i][3] + k).item()
                            z = (f * idx[i][4] + l).item()
                            # don't put areas that are not blocks in the list if strict
                            if scale in strict and v[x, y, z] == 0:
                                continue
                            x -= offsets[oi][0]
                            y -= offsets[oi][1]
                            z -= offsets[oi][2]
                            if loc_out.get((x, y, z)) is None:
                                loc_out[(x, y, z)] = {}
                            if loc_out[(x, y, z)].get(scale) is None:
                                loc_out[(x, y, z)][scale] = [to_tags[idx[i][1]]]
                            else:
                                loc_out[(x, y, z)][scale].append(to_tags[idx[i][1]])
        return loc_out, toplevel_out

    return block_eater


def draw_results(b, model, opts, vis, wins=None):
    if wins is None:
        w0 = None
        w1 = None
    else:
        w0 = wins[0]
        w1 = wins[1]
    sig = nn.Sigmoid()
    x, y = prepare_variables(b, opts)
    yhat = model(x)
    N = x.size(0)
    Y = torch.zeros(N, 3, 16, 16)
    Yhat = torch.zeros(N, 3, 16, 16)
    for t in range(N):
        syhat = sig(yhat[0][t]).data.cpu()
        sy = y[0][t].data.cpu()
        l = torch.zeros(16)
        for s in range(16):
            l[s] = sy[:, s, :, :].sum()
        mv, mid = l.max(0)
        ly = sy[:, mid[0], :, :].contiguous()
        lyhat = syhat[:, :, mid[0], :].contiguous()
        z = ly.view(-1, opts.nclasses).sum(0)
        zv, zid = z.sort()
        yim = torch.zeros(3, 16, 16)
        yhatim = torch.zeros(3, 16, 16)
        for s in range(3):
            yim[s] = ly[:, :, zid[-s]]
            yhatim[s] = lyhat[zid[-s], :, :]
        Y[t] = yim
        Yhat[t] = yhatim
    yim = tv.utils.make_grid(Y, nrow=8, padding=2, normalize=False, range=None, scale_each=False)
    yhatim = tv.utils.make_grid(
        Yhat, nrow=8, padding=2, normalize=False, range=None, scale_each=False
    )
    w0 = vis.image(img=yim, win=w0)
    w1 = vis.image(img=yhatim, win=w1)
    return Y, Yhat, w0, w1


def save_model(net, opts, path):
    sd = net.state_dict()
    for s in sd:
        sd[s] = sd[s].cpu()
    torch.save([sd, opts, net.classes], path)


# todo warn on changed opts
def load_model(path, opts=None):
    m = torch.load(path)
    newopts = m[1]
    net = model.Net(newopts)
    net.load_state_dict(m[0])
    net.classes = m[2]
    return net, newopts


def prepare_variables(b, opts):
    if opts.cuda:
        X = Variable(b[0].long().cuda())
        Y = []
        for l in b[1]:
            Y.append(Variable(l.float().cuda()))
    else:
        X = Variable(b[0].long())
        Y = []
        for l in b[1]:
            Y.append(Variable(l.float()))
    return X, Y


def train_epoch(model, lfn, optimizer, DL, opts):
    er = 0
    count = 0
    for b in DL:
        # dt = opts.lr
        model.train()
        x, y = prepare_variables(b, opts)
        optimizer.zero_grad()
        yhat = model(x)
        loss = lfn(yhat, y)
        er = er + loss.data[0]
        count = count + 1
        loss.backward()
        optimizer.step()
    return er / count


def validate(model, DL, opts):
    # er = 0
    l = 0
    count = 0
    L = nn.NLLLoss()
    lsm = nn.LogSoftmax(dim=1)
    if opts.cuda == 1:
        lsm.cuda()
    for b in DL:
        model.eval()
        x = Variable(b[0])
        y = Variable(b[1])
        if opts.cuda == 1:
            x = x.cuda()
            y = y.cuda()
        z = model(x)
        yhat = lsm(z)
        loss = L(yhat, y)
        l = l + loss.data[0]
        count = count + 1
    return l / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=1, help="0 for cpu")
    parser.add_argument("--batchsize", type=int, default=64, help="batchsize")
    parser.add_argument(
        "--epochsize", type=int, default=1000, help="number of examples in an epoch"
    )
    parser.add_argument("--nepoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("--sidelength", type=int, default=32, help="size of cube")
    parser.add_argument("--hidden_dim", type=int, default=64, help="size of hidden dim")
    parser.add_argument("--embedding_dim", type=int, default=8, help="size of blockid embedding")
    parser.add_argument("--lr", type=float, default=0.1, help="step size for net")
    parser.add_argument("--mom", type=float, default=0.0, help="momentum")
    parser.add_argument("--save_model", default="", help="where to save model (nowhere if blank)")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    opts = parser.parse_args()
    opts.nclasses = len(sdata.ALLLABELS)

    def init_fn(wid):
        np.random.seed(torch.initial_seed() % (2 ** 32))

    train_data = sdata.shape_dataset(nexamples=opts.epochsize)
    rDL = torch.utils.data.DataLoader(
        train_data,
        batch_size=opts.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=opts.num_workers,
        worker_init_fn=init_fn,
    )
    net = model.Net(opts)
    net.classes = sdata.LABELDICT
    lfn = model.multiscale_bcewl()
    if opts.cuda == 1:
        net.cuda()
        lfn.cuda()

    #   optimizer = optim.SGD(net.parameters(), lr = opts.lr, momentum = opts.mom)
    #    optimizer = optim.Adagrad(net.parameters(), lr = opts.lr)
    optimizer = optim.Adam(net.parameters(), lr=opts.lr, betas=(0.9, 0.999))

    ir = iter(rDL)
    b = ir.next()
    vis = visdom.Visdom(server="http://localhost")
    _, _, w0, w1 = draw_results(b, net, opts, vis)
    for i in range(opts.nepoch):
        er = train_epoch(net, lfn, optimizer, rDL, opts)
        print("train loss ", er, i)
        # l = validate(net, vDL, opts)
        # print('val loss ', l)
        if opts.save_model != "":
            save_model(net, opts, opts.save_model)
        ir = iter(rDL)
        b = ir.next()
        _, _, w0, w1 = draw_results(b, net, opts, vis, wins=[w0, w1])

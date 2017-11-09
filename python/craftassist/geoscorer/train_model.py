import numpy as np

# import time
import argparse
import torch
import torch.optim as optim
import models as model
import visdom


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
    X = b.long()
    if opts.cuda:
        X = X.cuda()
    return X


def train_epoch(model, lfn, optimizer, DL, opts):
    er = 0
    count = 0
    dlit = iter(DL)
    model.train()
    for i in range(len(DL)):
        b = dlit.next()
        # dt = opts.lr
        x = prepare_variables(b, opts)
        optimizer.zero_grad()
        scores = model(x)
        loss = lfn(scores)
        loss.backward()
        optimizer.step()
        er = er + loss.detach().item()
        count = count + 1
    return er / count


if __name__ == "__main__":
    # TODO different kinds of negatives...
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=1, help="0 for cpu")
    parser.add_argument("--batchsize", type=int, default=64, help="batchsize")
    parser.add_argument("--nneg", type=int, default=5, help="number of negs per pos")
    parser.add_argument("--dataset", default="shapes", help="shapes/segments/both")
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
    parser.add_argument(
        "--visualize", type=int, default=0, help="0 for nothing, 1 for visdom, 2 to render to file"
    )
    parser.add_argument(
        "--fixed_data",
        type=int,
        default=-1,
        help="if greater than zero use that many total examples",
    )
    opts = parser.parse_args()
    viz = visdom.Visdom(server="http://localhost")
    if opts.visualize > 1:
        import plot_voxels

        SP = plot_voxels.SchematicPlotter(viz)

    if opts.fixed_data > 0:
        opts.batchsize = opts.fixed_data
        opts.num_workers = 1
        nexamples = -opts.fixed_data
    else:
        nexamples = opts.epochsize

    def init_fn(wid):
        np.random.seed(torch.initial_seed() % (2 ** 32))

    if opts.dataset == "shapes":
        import shape_dataset as sdata

        train_data = sdata.SegmentCenterShapeData(
            nexamples=nexamples, shift_max=10, nneg=opts.nneg
        )
    elif opts.dataset == "segments":
        import inst_seg_dataset as sdata

        train_data = sdata.SegmentCenterInstanceData(
            nexamples=nexamples, shift_max=10, nneg=opts.nneg
        )
    else:
        raise ("that dataset not  done yet")
    rDL = torch.utils.data.DataLoader(
        train_data,
        batch_size=opts.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=opts.num_workers,
        worker_init_fn=init_fn,
        collate_fn=lambda x: torch.cat(x),
    )
    net = model.Net(opts)

    #    lfn = model.rank_loss()
    lfn = model.reshape_nll(nneg=opts.nneg)
    if opts.cuda == 1:
        net.cuda()
        lfn.cuda()

    #   optimizer = optim.SGD(net.parameters(), lr = opts.lr, momentum = opts.mom)
    optimizer = optim.Adagrad(net.parameters(), lr=opts.lr)
    # optimizer = optim.Adam(net.parameters(), lr=opts.lr, betas=(0.9, 0.999))

    vis = visdom.Visdom(server="http://localhost")
    #    _, _, w0, w1 = draw_results(b, net, opts, vis)

    for i in range(opts.nepoch):
        er = train_epoch(net, lfn, optimizer, rDL, opts)
        print("train loss ", er, i)
        # l = validate(net, vDL, opts)
        # print('val loss ', l)
        if opts.save_model != "":
            save_model(net, opts, opts.save_model)
#        if opts.visualize > 1:
#            ir = iter(rDL)
#            b = ir.next()
#        _, _, w0, w1 = draw_results(b, net, opts, vis, wins=[w0, w1])

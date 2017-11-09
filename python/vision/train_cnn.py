import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision as tv

# copy pasting folderdataset for now, use pytorch.4? FIXME
import hacky_data as tvd

PERM = torch.randperm(256)


def draw_img(x, vis, win=None):
    r = np.arange(0, 256) / 128 - 1
    CMAP = np.stack((r, np.roll(r, 80), np.roll(r, 160)))
    X = np.zeros((3, x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(3):
                X[k, i, j] = CMAP[k, PERM[int(x[i, j])]]
    return vis.image(img=X, win=win)


def binloader(path, S=32, meta=False):
    block = np.fromfile(path, np.uint8)
    edge = int((len(block) / 2) ** 0.5)
    block = block.reshape((edge, edge, 2))
    fac = edge // S
    block = block[0:-1:fac, 0:-1:fac, :]
    if not meta:
        block = block[:, :, 0]
    img = torch.from_numpy(block)
    return img.long()


class ResNet(nn.Module):
    def __init__(self, block, layers, embedding_dim=3, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        #        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2])
        #        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # FIXME for pytorch 4
                nn.init.kaiming_normal(m.weight, mode="fan_out")
            #                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(2).mean(2)
        #        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Net(nn.Module):
    def __init__(self, opts):
        super(Net, self).__init__()
        try:
            embedding_dim = opts.embedding_dim
        except:
            embedding_dim = 3
        try:
            num_words = opts.num_words
        except:
            num_words = 256
        try:
            num_classes = opts.nclasses
        except:
            num_classes = 1000

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.ResNet = ResNet(
            tv.models.resnet.BasicBlock,
            [2, 2, 2, 2],
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        # FIXME for pytorch .4: embedding is better now
        szs = list(x.size())
        x = x.view(-1)
        z = self.embedding(x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.transpose(1, 3).contiguous()
        return self.ResNet(z)


def save_model(net, opts, path):
    sd = net.state_dict()
    for s in sd:
        sd[s] = sd[s].cpu()
    torch.save([sd, opts, net.classes], path)


# todo warn on changed opts
def load_model(path, opts=None):
    m = torch.load(path)
    newopts = m[1]
    net = Net(newopts)
    net.load_state_dict(m[0])
    net.classes = m[2]
    return net, newopts


# not using metas for now ....
# FIXME maybe?
def build_numpy_eater(net, opts):
    net = net.cpu()

    def f(array):
        if opts.imsize == 32:
            array = array[:, 0:-1:2, 0:-1:2, 0]
        array = array.astype("int64")
        x = torch.from_numpy(array).contiguous()
        y = net(Variable(x))
        #        with torch.no_grad():
        #            y = net(Variable(torch.from_numpy(array)))
        out = []
        y = y.data
        for i in range(array.shape[0]):
            m, midx = torch.max(y[i], 0)
            out.append([net.classes[int(midx)]])
        return out

    return f


def load_dataset(cat_path, data_dir, S):
    c = open(cat_path)
    cat_map = {}
    for l in c:
        w = l.split()
        cat_map[int(w[0])] = w[1]
    U = torch.LongTensor(1024, S, S, 2)
    y = []
    count = 0
    for i in cat_map:
        d = data_dir + str(i) + "/"
        dd = os.listdir(d)
        l = cat_map[i]
        for f in dd:
            if f[:5] == "block":
                block = np.fromfile(d + f, np.uint8)
                if U.size(0) < count + 1:
                    szs = list(U.size())
                    szs[0] = 2 * szs[0]
                    szs = torch.Size(szs)
                    U.resize_(szs)
                edge = int((len(block) / 2) ** 0.5)
                block = block.reshape((edge, edge, 2))
                fac = edge // S
                block = block[0:-1:fac, 0:-1:fac, :]
                U[count] = torch.from_numpy(block)
                y.append(l)
                count += 1
    U = U[:count]
    Y = torch.LongTensor(count)
    all_labels = set(y)
    lcount = 0
    l2i = {}
    i2l = []
    for l in all_labels:
        l2i[l] = lcount
        lcount += 1
        i2l.append(l)
    for i in range(count):
        Y[i] = l2i[y[i]]

    return U, Y, {"i2l": i2l, "l2i": l2i}


def train_epoch(model, optimizer, DL, opts):
    er = 0
    count = 0
    L = nn.NLLLoss()
    lsm = nn.LogSoftmax(dim=1)
    if opts.cuda == 1:
        lsm.cuda()
    for b in DL:
        # dt = opts.lr
        model.train()
        x = Variable(b[0])
        y = Variable(b[1])
        if opts.cuda == 1:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        z = model(x)
        yhat = lsm(z)
        loss = L(yhat, y)
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
    parser.add_argument("--nepoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("--imsize", type=int, default=64, help="imsize, use 32 or 64")
    parser.add_argument("--embedding_dim", type=int, default=8, help="size of blockid embedding")
    parser.add_argument("--lr", type=float, default=0.1, help="step size for net")
    parser.add_argument("--mom", type=float, default=0.0, help="momentum")
    parser.add_argument(
        "--data_type",
        default="tensor",
        help="tensor for single big tensor, folder for DatasetFolder",
    )
    parser.add_argument("--data_dir", default="/scratch/jsgraymc/", help="path to data")
    parser.add_argument("--save_model", default="", help="where to save model (nowhere if blank)")
    opts = parser.parse_args()

    if opts.data_type == "tensor":
        cat_map = opts.data_dir + "cat_map.out"
        dl = opts.data_dir + "dl/"
        U, Y, ldict = load_dataset(cat_map, dl, opts.imsize)
        nclasses = len(ldict["i2l"])
        train_data = torch.utils.data.TensorDataset(U[:25000, :, :, 0].clone(), Y[:25000])
        val_data = torch.utils.data.TensorDataset(U[25000:, :, :, 0].clone(), Y[25000:])
    else:
        train_data = tvd.DatasetFolder(
            opts.data_dir + "train/",
            lambda path: binloader(path, S=opts.imsize, meta=False),
            [".bin"],
        )
        # fixme make a val set
        val_data = tvd.DatasetFolder(
            opts.data_dir + "val/",
            lambda path: binloader(path, S=opts.imsize, meta=False),
            [".bin"],
        )
        nclasses = len(train_data.classes)
    opts.nclasses = nclasses

    rDL = torch.utils.data.DataLoader(
        train_data,
        batch_size=opts.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    vDL = torch.utils.data.DataLoader(
        val_data,
        batch_size=opts.batchsize,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=4,
    )
    net = Net(opts)
    net.classes = train_data.classes
    if opts.cuda == 1:
        net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=opts.lr, momentum=opts.mom)

    for i in range(opts.nepoch):
        er = train_epoch(net, optimizer, rDL, opts)
        print("train loss ", er, i)
        l = validate(net, vDL, opts)
        print("val loss ", l)
        if opts.save_model != "":
            save_model(net, opts, opts.save_model)

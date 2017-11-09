import torch
import torch.nn as nn
import torch.optim as optim


class conv_body(nn.Module):
    def __init__(self, opts):
        super(conv_body, self).__init__()
        if not hasattr(opts, "nonlin"):
            opts.nonlin = "elu"
            opts.conv_body_layers = 3
            opts.use_batchnorm = 0
        if opts.nonlin == "elu":
            nonlin = nn.ELU
        else:
            nonlin = nn.ReLU
        edim = opts.embedding_dim
        self.edim = edim
        self.embedding = nn.Embedding(opts.n_words, edim, padding_idx=opts.n_words - 1)
        conv_list = []
        for i in range(opts.conv_body_layers):
            conv_list.append(nn.Conv1d(edim, edim, 5, padding=2))
            conv_list.append(nonlin(inplace=True))
            if opts.use_batchnorm == 1:
                conv_list.append(nn.BatchNorm1d(edim))
        self.convs = nn.Sequential(*conv_list)

    def forward(self, input):
        emb = self.embedding(input)
        emb = emb.transpose(1, 2).contiguous()
        return self.convs(emb)


# this is super wasteful in the sense the rnn is run for
# every subsequence
class recurrent_context_body(nn.Module):
    def __init__(self, opts):
        super(recurrent_context_body, self).__init__()
        edim = opts.embedding_dim
        self.opts = opts
        self.edim = edim
        self.embedding = nn.Embedding(opts.n_keys, edim, padding_idx=opts.null_key)
        self.gru = nn.GRU(
            opts.embedding_dim, opts.embedding_dim, self.opts.recurrent_context, batch_first=True
        )

    def forward(self, input):
        b, m = input.size()
        H = self.gru(self.embedding(input))
        return H[0]


# input is B x opts.max_depth LongTensor
class context_body(nn.Module):
    def __init__(self, opts):
        super(context_body, self).__init__()
        edim = opts.embedding_dim
        self.edim = edim
        self.depth_embedding = torch.nn.Parameter(torch.randn(opts.max_depth, edim))
        self.depth_embedding.requires_grad = True
        self.embedding = nn.Embedding(opts.n_keys, edim, padding_idx=opts.null_key)

    def forward(self, input):
        b, m = input.size()
        emb = self.embedding(input)
        emb = emb * self.depth_embedding.unsqueeze(0).expand(b, m, self.edim)
        return emb.mean(1).contiguous()


class simple_node_head(nn.Module):
    def __init__(self, opts, nout=13, node_type="key"):
        super(simple_node_head, self).__init__()
        self.node_type = node_type
        self.conv = nn.Conv1d(2 * opts.embedding_dim, opts.head_dim, 5, padding=2)
        if not hasattr(opts, "nonlin") or opts.nonlin == "elu":
            nonlin = nn.ELU
        else:
            nonlin = nn.ReLU
        self.nonlin = nonlin(inplace=True)
        self.use_bn = opts.use_batchnorm == 1
        if self.use_bn:
            self.bn = nn.BatchNorm1d(opts.head_dim)
        self.label_linear = nn.Linear(opts.head_dim, nout)
        if node_type == "intent" or node_type == "str":
            self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, input):
        emb = torch.cat((input[0], input[1].unsqueeze(2).expand_as(input[0])), 1)
        emb = self.conv(emb)
        emb = self.nonlin(emb)
        if self.use_bn:
            emb = self.bn(emb)
        # max instead?
        emb = emb.mean(2)
        out = self.label_linear(emb)
        if hasattr(self, "lsm"):
            out = self.lsm(out)
        return out


# concats the context to the start and end vector, mixes with a linear
# dot against conv of input with context appended
class simple_span_head(nn.Module):
    def __init__(self, opts):
        super(simple_span_head, self).__init__()
        self.embedding_dim = opts.embedding_dim
        self.span_head_dim = opts.span_head_dim
        # self.conv = nn.Conv1d(opts.embedding_dim, opts.span_head_dim, 5, padding = 2)
        self.conv = nn.Conv1d(2 * opts.embedding_dim, opts.span_head_dim, 5, padding=2)
        if not hasattr(opts, "nonlin") or opts.nonlin == "elu":
            nonlin = nn.ELU
        else:
            nonlin = nn.ReLU
        self.nonlin = nonlin(inplace=True)
        #        self.use_bn = (opts.use_batchnorm == 1)
        #        if self.use_bn:
        #            self.bn = nn.BatchNorm1d(opts.span_head_dim)

        self.start_linear = nn.Linear(2 * opts.embedding_dim, opts.span_head_dim)
        self.start = nn.Parameter(torch.randn(opts.embedding_dim))
        #        self.start = self.start/self.start.norm()
        self.start.requires_grad = True

        self.end_linear = nn.Linear(2 * opts.embedding_dim, opts.span_head_dim)
        self.end = nn.Parameter(torch.randn(opts.embedding_dim))
        #        self.end = self.end/self.end.norm()
        self.end.requires_grad = True

        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, input):
        # (B x n x opts.embedding_dim, B x opts.embedding_dim)
        #        emb = input[0]
        emb = torch.cat((input[0], input[1].unsqueeze(2).expand_as(input[0])), 1)
        emb = self.nonlin(self.conv(emb))
        #        if self.use_bn:
        #            emb = self.bn(emb)
        n = emb.size(2)
        B = emb.size(0)

        start = self.start.unsqueeze(0).expand(B, self.embedding_dim)
        start = self.start_linear(torch.cat((start, input[1]), dim=1))
        start = start.unsqueeze(2).expand(B, self.span_head_dim, n).contiguous()
        start_dot = (emb * start).sum(1)
        start_out = self.lsm(start_dot)

        end = self.end.unsqueeze(0).expand(B, self.embedding_dim)
        end = self.end_linear(torch.cat((end, input[1]), dim=1))
        end = end.unsqueeze(2).expand(B, self.span_head_dim, n).contiguous()
        end_dot = (emb * end).sum(1)
        end_out = self.lsm(end_dot)

        return start_out, end_out


# TODO make this into an nn.Module already :(
class TTADNet(object):
    def __init__(self, opts, dictionaries=None, path=""):
        if path != "":
            self.from_file = True
            self.sds = torch.load(path)
            self.opts = self.sds["opts"]
            self.intent_dict = self.sds.get("intent_dict")
            self.key_dict = self.sds["key_dict"]
            self.word_dict = self.sds["word_dict"]
        else:
            self.from_file = False
            if dictionaries is not None:
                self.intent_dict = dictionaries["intent_dict"]
                self.key_dict = dictionaries["key_dict"]
                self.word_dict = dictionaries["word_dict"]
            else:
                Exception("either input dictionaries or filename to build model")
            self.opts = opts
        self.net = {}
        net = {}
        net["body"] = conv_body(self.opts)
        if self.opts.recurrent_context > 0:
            net["context"] = recurrent_context_body(self.opts)
        else:
            net["context"] = context_body(self.opts)
        net["key_head"] = simple_node_head(
            self.opts, node_type="key", nout=len(self.key_dict["all_keys"]["i2w"])
        )
        net["intent_head"] = simple_node_head(self.opts, node_type="intent", nout=13)
        net["str_head"] = simple_node_head(
            self.opts, node_type="str", nout=len(self.key_dict["str_vals"]["i2w"])
        )
        if not hasattr(self.opts, "span_head_dim"):
            self.opts.span_head_dim = self.opts.head_dim  # backwards compatibility
        net["span_head"] = simple_span_head(self.opts)

        if self.from_file:
            net["body"].load_state_dict(self.sds["body"])
            net["context"].load_state_dict(self.sds["context"])
            net["str_head"].load_state_dict(self.sds["str_head"])
            net["key_head"].load_state_dict(self.sds["key_head"])
            net["span_head"].load_state_dict(self.sds["span_head"])
            net["intent_head"].load_state_dict(self.sds["intent_head"])
        self.net = net

    def save(self, target_path):
        self.cpu()
        sds = {}
        sds["opts"] = self.opts
        sds["intent_dict"] = self.intent_dict
        sds["key_dict"] = self.key_dict
        sds["word_dict"] = self.word_dict
        sds["body"] = self.net["body"].state_dict()
        sds["context"] = self.net["context"].state_dict()
        sds["key_head"] = self.net["key_head"].state_dict()
        sds["str_head"] = self.net["str_head"].state_dict()
        sds["span_head"] = self.net["span_head"].state_dict()
        sds["intent_head"] = self.net["intent_head"].state_dict()
        torch.save(sds, target_path)
        if self.opts.cuda:
            self.cuda()

    def embed_words(self, x):
        return self.net["body"](x)

    def embed_context(self, x, lengths=None):
        H = self.net["context"](x)
        if lengths is None or H.dim() == 2:
            return H
        else:
            r = torch.arange(H.size(0), dtype=x.dtype, device=x.device)
            return H[r, lengths]

    def embed_words_and_context(self, w, c, lengths=None):
        return [self.embed_words(w), self.embed_context(c)]

    def get_intent(self, h_w, h_c):
        return self.net["intent_head"]([h_w, h_c])

    def get_key(self, h_w, h_c):
        return self.net["key_head"]([h_w, h_c])

    def get_str(self, h_w, h_c):
        return self.net["str_head"]([h_w, h_c])

    def get_span(self, h_w, h_c):
        return self.net["span_head"]([h_w, h_c])

    def forward(self, words, context, context_len=None, mode="all"):
        h_w = self.embed_words(words)
        h_c = self.embed_context(context, lengths=context_len)
        if mode == "key":
            return self.get_key(h_w, h_c)
        if mode == "str":
            return self.get_str(h_w, h_c)
        elif mode == "intent":
            return self.get_intent(h_w, h_c)
        elif mode == "span":
            return self.get_span(h_w, h_c)
        else:  # all
            return (
                self.get_key(h_w, h_c),
                self.get_str(h_w, h_c),
                self.get_intent(h_w, h_c),
                self.get_span(h_w, h_c),
            )

    def eval(self):
        for part, net in self.net.items():
            net.eval()

    def train(self):
        for part, net in self.net.items():
            net.train()

    def cuda(self):
        for part, net in self.net.items():
            net.cuda()

    def cpu(self):
        for part, net in self.net.items():
            net.cpu()


PWEIGHT = 10


class MarginHinge(nn.Module):
    def __init__(self, margin):
        super(MarginHinge, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # this takes 0,1 --> -1,1
        u = 2 * y - 1
        #        pmask = PWEIGHT*(u > 0).type_as(u)
        #        nmask = (u < 0).type_as(u)
        #        mask = pmask + nmask
        return self.relu(self.margin - x * u)


#        return self.relu(self.margin - x*u)**2
#        return(mask*self.relu(self.margin - x*u))


def build_optimizers(opts, model):
    lrs = {
        "context": opts.lr_context,
        "body": opts.lr_body,
        "key_head": opts.lr_key_head,
        "str_head": opts.lr_str_head,
        "intent_head": opts.lr_intent_head,
        "span_head": opts.lr_span_head,
    }

    optimizers = {}
    if opts.optimizer == "sgd":
        for part, net in model.net.items():
            optimizers[part] = optim.SGD(
                net.parameters(), lr=lrs[part], momentum=opts.mom, weight_decay=opts.wd
            )
    elif opts.optimizer == "adagrad":
        for part, net in model.net.items():
            optimizers[part] = optim.Adagrad(net.parameters(), lr=lrs[part], weight_decay=opts.wd)

    elif opts.optimizer == "adam":
        for part, net in model.net.items():
            optimizers[part] = optim.Adam(net.parameters(), lr=lrs[part], weight_decay=opts.wd)

    else:
        Exception("unknown optimizer type, use either adam, adagrad, or sgd")

    return optimizers

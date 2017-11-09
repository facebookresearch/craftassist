"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
import models
from geoscorer_util import *


def train_epoch(model, lfn, optimizer, DL, opts):
    l = 0
    error = 0
    count = 0
    dlit = iter(DL)
    model.train()
    for i in range(len(DL)):
        b = dlit.next()
        x = models.prepare_variables(b, opts)
        optimizer.zero_grad()
        scores = model(x)
        error += scores.view(-1, opts.nneg + 1).argmax(dim=1).ne(0).sum().detach().item()
        loss = lfn(scores)
        loss.backward()
        optimizer.step()
        l = l + loss.detach().item()
        count = count + 1
    return (l / count, error / (count * opts.batchsize))


if __name__ == "__main__":
    parser = get_base_train_parser()
    parser.add_argument("--nneg", type=int, default=5, help="number of negs per pos")
    parser.add_argument("--loss", type=str, default="nll", help="loss type (nll|rank)")
    opts = parser.parse_args()

    # Set the number of examples per epoch
    if opts.fixed_data > 0:
        opts.batchsize = opts.fixed_data
        opts.num_workers = 1
        nexamples = -opts.fixed_data
    else:
        nexamples = opts.epochsize

    # Get the loss function module
    if opts.loss == "rank":
        lfn = models.rank_loss()
    elif opts.loss == "nll":
        lfn = models.reshape_nll(nneg=opts.nneg)
    else:
        raise Exception("Loss type {} undefined".format(opts.loss))

    # Get the dataset and dataloader
    if opts.dataset == "shapes":
        import shape_dataset as sdata

        train_data = sdata.SegmentContextCombinedShapeData(
            nexamples=nexamples, shift_max=10, nneg=opts.nneg
        )
    elif opts.dataset == "segments":
        import inst_seg_dataset as sdata

        train_data = sdata.SegmentContextCombinedInstanceData(
            nexamples=nexamples, shift_max=10, nneg=opts.nneg
        )
    else:
        raise Exception("Dataset {} undefined".format(opts.dataset))
    train_dataloader = get_dataloader(
        dataset=train_data, opts=opts, collate_fxn=lambda x: torch.cat(x)
    )

    # Setup the model and optimizer
    model = models.ValueNet(opts)
    model_dict = {"value_model": model}
    if opts.cuda == 1:
        to_cuda([model, lfn])
    optimizer = models.get_optim(model.parameters(), opts)

    # Load checkpoint if it exists
    if opts.checkpoint != "":
        models.load_checkpoint(model_dict, optimizer, opts.checkpoint, opts)
    else:
        models.check_and_print_opts(opts, None)

    # Run the train loop
    for i in range(opts.nepoch):
        train_loss, train_error = train_epoch(model, lfn, optimizer, train_dataloader, opts)
        pretty_log(
            "train loss {:<5.4f} error {:<5.2f} {}".format(train_loss, train_error * 100, i)
        )
        if opts.checkpoint != "":
            metadata = {"epoch": i, "train_loss": train_loss, "train_error": train_error}
            models.save_checkpoint(model_dict, metadata, opts, optimizer, opts.checkpoint)

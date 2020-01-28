"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
import sys
import os
import time

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../")
sys.path.append(CRAFTASSIST_DIR)
VOXEL_MODELS_DIR = os.path.join(GEOSCORER_DIR, "../../")
sys.path.append(VOXEL_MODELS_DIR)

from geoscorer_util import *
import combined_dataset as cd
import models


def set_modules(tms, train):
    for m in ["context_net", "seg_net", "score_module", "seg_direction_net"]:
        if m not in tms:
            continue
        if train:
            tms[m].train()
        else:
            tms[m].eval()


def get_scores_from_datapoint(tms, data, opts):
    contexts = models.prepare_variables(data[0], opts)
    segments = models.prepare_variables(data[1], opts)
    targets = models.prepare_variables(data[2].squeeze(1), opts)
    tms["optimizer"].zero_grad()

    cont_dir = opts.get("cont_use_direction", False) or opts.get(
        "cont_use_xyz_from_viewer_look", False
    )

    if cont_dir:
        viewer_pos = models.prepare_variables(data[3], opts)
        viewer_look = models.prepare_variables(data[4], opts)
        dir_vec = models.prepare_variables(data[5], opts)
        c_embeds = tms["context_net"]([contexts, viewer_pos, viewer_look, dir_vec])
    else:
        c_embeds = tms["context_net"]([contexts])
    s_embeds = tms["seg_net"](segments)

    if opts.get("seg_direction_net", False):
        viewer_pos = models.prepare_variables(data[3], opts)
        viewer_look = models.prepare_variables(data[4], opts)
        dir_vec = models.prepare_variables(data[5], opts)
        s_embeds = s_embeds.squeeze()
        # Add back in the batch dim for batch size 1
        if s_embeds.dim() < 2:
            s_embeds = s_embeds.unsqueeze(0)
        s_embeds = tms["seg_direction_net"]([s_embeds, viewer_pos, viewer_look, dir_vec])

    scores = tms["score_module"]([c_embeds, s_embeds])
    return targets, scores


def train_epoch(tms, DL, opts):
    l = 0
    error = 0
    count = 0
    dlit = iter(DL)
    set_modules(tms, train=True)
    for i in range(len(DL)):
        b = dlit.next()
        targets, scores = get_scores_from_datapoint(tms, b, opts)
        loss = tms["lfn"]([scores, targets])
        max_ind = torch.argmax(scores, dim=1)
        num_correct = sum(max_ind.eq(targets)).item()
        error += opts["batchsize"] - num_correct
        loss.backward()
        tms["optimizer"].step()
        l = l + loss.detach().item()
        count = count + 1
    return (l / count, error / (count * opts["batchsize"]))


def run_visualization(
    sp, tms, dataset, opts, checkpoint_path=None, num_examples=2, tsleep=1, loadpath=False
):
    if loadpath and checkpoint_path is not None and checkpoint_path != "":
        new_tms = models.load_context_segment_checkpoint(
            checkpoint_path, opts, backup=False, verbose=True
        )
    else:
        new_tms = tms

    set_modules(new_tms, train=False)
    for n in range(num_examples):
        # Visualization
        b = dataset[n]
        context = b[0]
        seg = b[1]
        target = b[2]
        target_coord = index_to_coord(target.item(), 32)
        completed_shape = combine_seg_context(seg, context, target_coord, seg_mult=3)
        sp.drawPlotly(context)
        sp.drawPlotly(seg)
        sp.drawPlotly(completed_shape)
        b = [t.unsqueeze(0) for t in b]
        targets, scores = get_scores_from_datapoint(new_tms, b, opts)
        max_ind = torch.argmax(scores, dim=1)

        # convert back to single voxel grids and complete shape
        context = context.squeeze(0)
        seg = seg.squeeze(0)
        predicted_coord = index_to_coord(max_ind.item(), 32)
        predicted_shape = combine_seg_context(seg, context, predicted_coord, seg_mult=3)
        sp.drawPlotly(predicted_shape)
    time.sleep(tsleep)


def parse_dataset_ratios(opts):
    ratios_str = opts["dataset_ratios"]
    ratio = {}
    try:
        l_s = ratios_str.split(",")
        print("\n>> Using datasets in the following proportions:")
        for t in l_s:
            name, prob = t.split(":")
            ratio[name] = float(prob)
            print("  -     {}: {}".format(name, prob))
        print("")
    except:
        raise Exception("Failed to parse the dataset ratio string {}".format(ratios_str))
    return ratio


def setup_dataset_and_loader(opts):
    extra_params = {
        "min_seg_size": opts.get("min_seg_size", 6),
        "use_saved_data": opts.get("use_saved_data", False),
        "fixed_cube_size": opts.get("fixed_cube_size", None),
        "fixed_center": opts.get("fixed_center", False),
    }
    dataset = cd.SegmentContextSeparateData(
        nexamples=opts["epochsize"],
        useid=opts["useid"],
        extra_params=extra_params,
        ratios=parse_dataset_ratios(opts),
    )
    dataloader = get_dataloader(dataset=dataset, opts=opts, collate_fxn=multitensor_collate_fxn)
    return dataset, dataloader


if __name__ == "__main__":
    parser = get_base_train_parser()
    add_directional_flags(parser)
    add_dataset_flags(parser)
    parser.add_argument(
        "--backup", type=bool, default=False, help="backup the checkpoint path before saving to it"
    )
    parser.add_argument(
        "--visualize_epochs", type=bool, default=False, help="use visdom to visualize progress"
    )
    opts = vars(parser.parse_args())

    # Setup the data, models and optimizer
    dataset, dataloader = setup_dataset_and_loader(opts)
    tms = models.get_context_segment_trainer_modules(
        opts, opts["checkpoint"], backup=opts["backup"], verbose=True
    )
    if opts["cuda"] == 1:
        # The context and seg net were already moved
        to_cuda([tms["score_module"], tms["lfn"]])

    # Setup visualization
    sp = None
    if opts["visualize_epochs"]:
        import visdom
        import plot_voxels as pv

        vis = visdom.Visdom(server="http://localhost")
        sp = pv.SchematicPlotter(vis)
        run_visualization(sp, tms, dataset, opts, None, 2, 1, False)

    # Run training
    for i in range(opts["nepoch"]):
        train_loss, train_error = train_epoch(tms, dataloader, opts)
        pretty_log(
            "train loss {:<5.4f} error {:<5.2f} {}".format(train_loss, train_error * 100, i)
        )
        if opts["checkpoint"] != "":
            metadata = {"epoch": i, "train_loss": train_loss, "train_error": train_error}
            models.save_checkpoint(tms, metadata, opts, opts["checkpoint"])
        if opts["visualize_epochs"]:
            run_visualization(sp, tms, dataset, opts, opts["checkpoint"], 2, 1, False)

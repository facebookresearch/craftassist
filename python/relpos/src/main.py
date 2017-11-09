import sys
import os
import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# from IPython import embed

from config import RANK_NORM_FACTOR, DEBUG
from config import Config
from utils import RelposData
from model import BaselineNN


def train(config, model, optimizer, data):
    """
        Train function.
        Return (model, optimizer, avg_loss).
    """
    model.train()

    avg_loss = 0
    total = 0
    for batch_idx, batch_data in tqdm(
        enumerate(data.loader), desc="Training", total=len(data.loader)
    ):
        if DEBUG:
            if batch_idx > 100:
                break
        batch_size = batch_data[0].size(0)
        total += batch_size
        assert batch_data[1].size(0) == batch_size
        boxsize = config.args.boxsize

        positive_data, negative_data = data.process_data_batch(batch_data)
        positive_data = autograd.Variable(positive_data)
        negative_data = autograd.Variable(negative_data)
        if config.use_cuda:
            positive_data, negative_data = positive_data.cuda(), negative_data.cuda()
        num_neg = negative_data.size(1)

        # TODO(demiguo): check if this loss makes sense
        positive_scores = model.get_score(positive_data)  # normalized
        negative_scores = model.get_score(
            negative_data.contiguous().view(batch_size * num_neg, boxsize, boxsize, boxsize)
        )
        positive_scores = (
            positive_scores.contiguous().view(batch_size, 1).expand(batch_size, num_neg)
        )
        negative_scores = negative_scores.contiguous().view(batch_size, num_neg)

        loss = negative_scores - positive_scores + config.args.delta_constant
        if DEBUG:
            print("positive_loss[0]=", positive_scores[0])
            print("negative_scores[0]=", negative_scores[0])
            print("loss[0] 1st=", loss[0])
        assert loss.size() == (batch_size, num_neg)
        loss = torch.max(loss, dim=1)[0].view(batch_size, 1)
        if DEBUG:
            print("loss[0] 2nd=", loss[0])
        loss = (loss > 0).float() * loss
        if DEBUG:
            print("loss[0] final", loss[0])
        loss = torch.mean(loss)
        avg_loss += loss.item() * batch_size
        if DEBUG:
            print("avg_loss=%.3lf" % avg_loss / total)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss /= total
    return model, optimizer, avg_loss


# TODO(demiguo): think more about evaluation methods:
def evaluate(config, model, data):
    """
        Evaluate function.
        Return (avg_loss, avg_norm_rank).
    """
    model.eval()

    avg_loss = 0
    avg_norm_rank = (
        0
    )  # normalized rank ... assume there are RANK_NORM_FACTOR positive/negative examples
    total = 0
    for batch_idx, batch_data in tqdm(
        enumerate(data.loader), desc="Evaluating", total=len(data.loader)
    ):
        batch_size = batch_data[0].size(0)
        total += batch_size
        assert batch_data[1].size(0) == batch_size
        boxsize = config.args.boxsize

        positive_data, negative_data = data.process_data_batch(batch_data)
        positive_data = autograd.Variable(positive_data)
        negative_data = autograd.Variable(negative_data)
        if config.use_cuda:
            positive_data, negative_data = positive_data.cuda(), negative_data.cuda()
        num_neg = negative_data.size(1)

        # TODO(demiguo): check if this loss makes sense
        positive_scores = model.get_score(positive_data)  # normalized
        negative_scores = model.get_score(
            negative_data.contiguous().view(batch_size * num_neg, boxsize, boxsize, boxsize)
        )
        positive_scores = (
            positive_scores.contiguous().view(batch_size, 1).expand(batch_size, num_neg)
        )
        negative_scores = negative_scores.contiguous().view(batch_size, num_neg)

        loss = negative_scores - positive_scores + config.args.delta_constant
        assert loss.size() == (batch_size, num_neg)
        loss = torch.max(loss, dim=1)[0].view(batch_size, 1)
        loss = (loss > 0).float() * loss
        if DEBUG:
            print("negative_scores[0]=", negative_scores[0])
            print("positive_scores[0]=", positive_scores[0])
            print("loss[0]=", loss[0])
        loss = torch.mean(loss)
        avg_loss += loss.item() * batch_size

        positive_scores = positive_scores.cpu().data.numpy()
        negative_scores = negative_scores.cpu().data.numpy()
        for b in range(batch_size):
            scores = negative_scores[b]
            assert scores.shape[0] == num_neg
            scores = np.append(scores, positive_scores[b][0])
            assert scores.shape[0] == num_neg + 1  # index num_neg represents positive example
            ranks = np.argsort(-scores)  # the higher score, the better
            prank = list(ranks).index(num_neg) + 1  # rank of the positive example, 1-indexed
            avg_norm_rank += prank / (num_neg + 1.0) * RANK_NORM_FACTOR

    avg_loss /= total
    avg_norm_rank /= total
    return avg_loss, avg_norm_rank


def visualize(config, model, data, visual_name="default", n_sample=1, neg_cap=10):
    """
        Visualization function.
    """
    model.eval()
    counter = 0
    for batch_data in data.loader:
        batch_size = batch_data[0].size(0)
        assert batch_data[1].size(0) == batch_size
        boxsize = config.args.boxsize

        positive_data, negative_data, positive_global, negative_global, positive_local, negative_local, final_schematic = data.process_data_batch(
            batch_data, visualize=True
        )
        positive_data = autograd.Variable(positive_data)
        negative_data = autograd.Variable(negative_data)
        if config.use_cuda:
            positive_data, negative_data = positive_data.cuda(), negative_data.cuda()
        num_neg = negative_data.size(1)

        # TODO(demiguo): check if this loss makes sense
        positive_scores = model.get_score(positive_data)  # normalized
        negative_scores = model.get_score(
            negative_data.contiguous().view(batch_size * num_neg, boxsize, boxsize, boxsize)
        )
        positive_scores = (
            positive_scores.contiguous().view(batch_size, 1).expand(batch_size, num_neg)
        )
        negative_scores = negative_scores.contiguous().view(batch_size, num_neg)

        positive_scores = positive_scores.cpu().data.numpy()
        negative_scores = negative_scores.cpu().data.numpy()
        for b in range(batch_size):
            if counter == n_sample:
                return
            counter += 1
            scores = negative_scores[b]
            assert scores.shape[0] == num_neg
            scores = np.append(scores, positive_scores[b][0])
            assert scores.shape[0] == num_neg + 1  # index num_neg represents positive example
            ranks = np.argsort(-scores)  # the higher score, the better

            config.args.visual_counter += 1
            cur_visual_dir = os.path.join(
                config.args.visual_dir, "%d-%s" % (config.args.visual_counter, visual_name)
            )
            if not os.path.exists(cur_visual_dir):
                os.makedirs(cur_visual_dir)

            neg_cnt = 0
            # visualize final house schematic
            np.save(os.path.join(cur_visual_dir, "final_schematic"), final_schematic[b])
            if "final" in config.args.vis_mode:
                cmd = (
                    "python ../../render_schematic.py %s/final_schematic.npy --out-dir=%s > tmp_visual_out 2>&1"
                    % (cur_visual_dir, cur_visual_dir)
                )
                os.system(cmd)

            # visualize positive/negative example schematic
            for e in tqdm(range(num_neg + 1), desc="visualize (%d/%d):" % (counter, n_sample)):
                new_dir_name = "rank%d_score%.3lf_%s" % (
                    e + 1,
                    scores[ranks[e]],
                    "pos" if ranks[e] == num_neg else "neg",
                )
                new_dir_name = os.path.join(cur_visual_dir, new_dir_name)
                if not os.path.exists(new_dir_name):
                    os.makedirs(new_dir_name)
                if ranks[e] == num_neg:
                    global_schematic = positive_global[b]
                    local_schematic = positive_local[b]
                else:
                    global_schematic = negative_global[b][ranks[e]]
                    local_schematic = negative_local[b][ranks[e]]
                    neg_cnt += 1
                global_dir = os.path.join(new_dir_name, "global")
                local_dir = os.path.join(new_dir_name, "local")
                os.makedirs(global_dir)
                os.makedirs(local_dir)
                np.save(os.path.join(global_dir, "global_schematic"), global_schematic)
                np.save(os.path.join(local_dir, "local_schematic"), local_schematic)
                if (
                    ranks[e] == num_neg or neg_cnt <= neg_cap
                ):  # only visualize at most neg_cap negative examples
                    if "global" in config.args.vis_mode:
                        cmd = (
                            "python ../../render_schematic.py %s/global_schematic.npy --out-dir=%s > tmp_visual_out 2>&1"
                            % (global_dir, global_dir)
                        )
                        os.system(cmd)
                    if "local" in config.args.vis_mode:
                        cmd = (
                            "python ../../render_schematic.py %s/local_schematic.npy --out-dir=%s > tmp_visual_out 2>&1"
                            % (local_dir, local_dir)
                        )
                        os.system(cmd)


if __name__ == "__main__":
    config = Config()
    config.log.info("=> Finish loading config ...")

    train_data = RelposData(config, config.train_source, True)
    config.log.info("=> Finish building Train ...")
    val_data = RelposData(config, config.val_source, False)
    config.log.info("=> Finish building Valid ...")
    test_data = RelposData(config, config.test_source, False)
    config.log.info("=> Finish building Test ...")

    model = BaselineNN(config)
    if config.use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(
        model.get_train_parameters(), lr=config.args.lr, weight_decay=config.args.wd
    )

    # load from checkpoint if specified
    def load_checkpoint(model, optimizer, checkpoint_file):
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            config.log.info("[fatal] No checkpoint found at {}".format(checkpoint_file))

    if config.args.load_model != "":
        config.log.info("=> Loading model from %s" % config.args.load_model)
        load_checkpoint(model, optimizer, config.args.load_model)

    if config.args.mode == "eval":
        train_loss, train_norm_rank = evaluate(config, model, train_data)
        val_loss, val_norm_rank = evaluate(config, model, val_data)
        test_loss, test_norm_rank = evaluate(config, model, test_data)
        config.log.info(
            "EVAL Train Loss: %.3lf | Train Normalized Rank: %.3lf" % (train_loss, train_norm_rank)
        )
        config.log.info(
            "EVAL Val Loss: %.3lf | Val Normalized Rank: %.3lf" % (val_loss, val_norm_rank)
        )
        config.log.info(
            "EVAL Test Loss: %.3lf | Test Normalized Rank: %.3lf" % (test_loss, test_norm_rank)
        )
        if config.args.vis_mode != "":
            visualize(config, model, val_data, "val", 5)
            visualize(config, model, train_data, "train", 5)
        sys.exit(0)
    elif config.args.mode == "vis":
        visualize(config, model, val_data, "val", 10)
        visualize(config, model, train_data, "train", 5)
        sys.exit(0)

    assert config.args.mode == "train", "now only support [eval, train] mode"

    best_val_loss = 10000000
    for epoch in range(config.args.epochs):
        model, optimizer, avg_loss = train(config, model, optimizer, train_data)
        val_loss, val_norm_rank = evaluate(config, model, val_data)
        # TODO(demiguo): evaluate function
        config.log.info("EPOCH[%d] Train Loss: %.3lf" % (epoch, avg_loss))
        config.log.info(
            "EPOCH[%d] Val Loss: %.3lf | Val Normalized Rank: %.3lf"
            % (epoch, val_loss, val_norm_rank)
        )

        def save_checkpoint():
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config.args": config.args,
            }
            checkpoint_file = os.path.join(config.args.model_dir, "epoch%d" % epoch)
            config.log.info("=> Saving to checkpoint @ epoch %d to %s" % (epoch, checkpoint_file))
            torch.save(checkpoint, checkpoint_file)

        if val_loss < best_val_loss:
            config.log.info(
                "=> At Epoch %d, update best validation loss: %.3lf (new) to %.3lf (old)"
                % (epoch, val_loss, best_val_loss)
            )
            best_val_loss = val_loss
            save_checkpoint()

            # evaluate on test
            test_loss, test_norm_rank = evaluate(config, model, test_data)
            config.log.info(
                "EPOCH[%d] Test Loss: %.3lf | Test Normalized Rank: %.3lf"
                % (epoch, test_loss, test_norm_rank)
            )

            # visualize
            if config.args.vis_mode != "":
                visualize(config, model, train_data, "train", 32)
                visualize(config, model, val_data, "val", 32)

""" Preprocessing """
import os
import argparse
import random
import pickle
import numpy as np
import glob
from IPython import embed
from tqdm import tqdm


def check_valid(occupancy, xyzs, tx, ty, tz):
    for xyz in xyzs:
        x, y, z = xyz
        x += tx
        y += ty
        z += tz
        if (
            x < occupancy.shape[0]
            and y < occupancy.shape[1]
            and z < occupancy.shape[2]
            and occupancy[x][y][z] == 1
        ):
            return False
    return True


def get_negative_examples_valid(occupancy, xyzs, at_most=-1):
    """
        Get negative examples translation vectors.
        Currently, we randomly sample from valid translation vectors.
        NB: There's bias since we ensure the minimum corner is in the house bounding box ...
            at inference we don't know about the bounding box ...
    """
    minx, miny, minz = np.min(np.array(xyzs), axis=0)
    negative_examples = []
    for tx in range(-minx, occupancy.shape[0] - minx):
        for ty in range(-miny, occupancy.shape[1] - miny):
            for tz in range(-minz, occupancy.shape[2] - minz):
                if tx == 0 and ty == 0 and tz == 0:
                    continue
                if check_valid(occupancy, xyzs, tx, ty, tz):
                    negative_examples.append((tx, ty, tz))
    if len(negative_examples) == 0:
        print("No Negative Examples")
        embed()
        return []
    random.shuffle(negative_examples)
    if at_most == -1:
        return negative_examples
    else:
        return negative_examples[: min(len(negative_examples), at_most)]


avg_rank_overlap = 0
avg_rank_dist = 0
avg_num_neg = 0
avg_num_neg_strict = 0
occupancy_sizes = []
total_examples = 0
ANALYZE = True


def get_num_overlap(occupancy, xyzs, tvec):
    dxyzs = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if abs(dx) + abs(dy) + abs(dz) == 1:
                    dxyzs.append((dx, dy, dz))

    def is_valid(x, y, z):
        return (
            x >= 0
            and y >= 0
            and z >= 0
            and x < occupancy.shape[0]
            and y < occupancy.shape[1]
            and z < occupancy.shape[2]
        )

    cnt = 0
    for xyz in xyzs:
        real_xyz = np.array(xyz) + np.array(tvec)
        for dxyz in dxyzs:
            next_xyz = np.array(dxyz) + real_xyz
            nx, ny, nz = next_xyz
            if is_valid(nx, ny, nz) and occupancy[nx][ny][nz] == 1:
                cnt += 1
    return cnt


def get_dist_to_cent(partial_cent, comp_cent, tvec):
    dist = np.array(comp_cent) + np.array(tvec) - np.array(partial_cent)
    return np.sum(np.abs(dist))


def get_centroid(xyzs):
    return tuple(np.mean(np.array(xyzs), axis=0))


def fits(occupancy, xyzs, tvec):
    maxx, maxy, maxz = occupancy.shape
    tx, ty, tz = tvec
    for xyz in xyzs:
        x, y, z = xyz
        x += tx
        y += ty
        z += tz
        if x < 0 or y < 0 or z < 0 or x >= maxx or y >= maxy or z >= maxz:
            return False
    return True


def get_negative_examples(occupancy, xyzs, at_most=-1):
    negative_examples = get_negative_examples_valid(occupancy, xyzs, at_most)
    if not ANALYZE:
        return negative_examples
    global avg_rank_overlap, avg_rank_dist
    global avg_num_neg, avg_num_neg_strict, occupancy_sizes
    global total_examples
    occupancy_sizes.append((occupancy.shape[0], occupancy.shape[1], occupancy.shape[2]))
    if ANALYZE:
        print("occupancy shape:", occupancy.shape)
    all_examples = [(0, 0, 0)]
    all_examples.extend(negative_examples)

    # sort by overlaps, distance to centroid of the partially built house,
    # and calculate average positive example rank
    # print out those values
    num_overlaps = []
    dist_to_cent = []

    o_xyzs = []
    for i in range(occupancy.shape[0]):
        for j in range(occupancy.shape[1]):
            for z in range(occupancy.shape[2]):
                o_xyzs.append((i, j, z))
    partial_cent = get_centroid(o_xyzs)
    comp_cent = get_centroid(xyzs)
    for example in all_examples:
        num_overlaps.append(get_num_overlap(occupancy, xyzs, example))
        dist_to_cent.append(get_dist_to_cent(partial_cent, comp_cent, example))
    num_overlaps_ids = np.argsort(-np.array(num_overlaps))
    dist_to_cent_ids = np.argsort(-np.array(dist_to_cent))
    print("overlaps:", num_overlaps)
    print("dist to cent:", dist_to_cent)
    rank_overlap = num_overlaps_ids.tolist().index(0) + 1
    rank_dist = dist_to_cent_ids.tolist().index(0) + 1
    avg_rank_overlap += rank_overlap
    avg_rank_dist += rank_dist
    print("rank overlap:%d rank dist:%d" % (rank_overlap, rank_dist))

    # calculate avg # of negative examples,
    # avg # of negative examples (fits all in the bounding box)
    avg_num_neg += len(negative_examples)
    num_neg_strict = 0
    for example in negative_examples:
        if fits(occupancy, xyzs, example):
            num_neg_strict += 1
    avg_num_neg_strict += num_neg_strict
    print(
        "total negative examples %d, num neg strict: %d "
        % (len(negative_examples), num_neg_strict)
    )

    total_examples += 1

    return negative_examples


def data_preprocess(load_folder, save_folder, debug=False):
    """
        Given components sorted by time heuristic,
            calculate corresponding centroid, positive/negative examples
    """
    raw_files = glob.glob(load_folder + "*.pkl")
    print("load %d raw data files ..." % len(raw_files))
    label_dict = {}
    preprocessed_data = []
    for i, raw_file in enumerate(raw_files):
        if debug and i >= 100:
            break
        raw_data = np.load(raw_file)

        # first find maxx, maxy, maxz
        inf = 10000
        maxx, maxy, maxz = -inf, -inf, -inf
        minx, miny, minz = inf, inf, inf
        if len(raw_data) == 0:
            continue
        for component in raw_data:
            label, xyzs = component
            for xyz in xyzs:
                x, y, z = xyz
                maxx = max(maxx, x)
                maxy = max(maxy, y)
                maxz = max(maxz, z)
                minx = min(minx, x)
                miny = min(miny, y)
                minz = min(minz, z)
        assert minx >= 0 and miny >= 0 and minz >= 0  # sanity check
        occupancy = np.zeros((maxx + 1, maxy + 1, maxz + 1), dtype=np.int32)

        # calculate centroid, positive/negative [transformation] examples
        house_data = []
        for component in tqdm(raw_data, desc="preprocess raw file (%d/%d)" % (i, len(raw_files))):
            label, xyzs = component
            # get label id
            if label not in label_dict:
                label_dict[label] = len(label_dict)
            label_id = label_dict[label]
            # calculate centroid of xyzs
            centroid = tuple(np.mean(np.array(xyzs), axis=0).astype(np.int32))
            # positive example translation vector
            positive_examples = [(0, 0, 0)]
            # negative example translation vectors
            negative_examples = get_negative_examples(occupancy, xyzs, 20)

            # update occupancy
            for xyz in xyzs:
                occupancy[x][y][z] = 1
            component_data = (
                label_id,
                centroid,
                positive_examples,
                negative_examples,
                component,
                raw_file,
            )
            house_data.append(component_data)
        preprocessed_data.append(house_data)

    # print out statistics
    if ANALYZE:
        print(
            "avg_rank_overlap:%.3lf, avg_rank_dist:%.3lf\n"
            % (1.0 * avg_rank_overlap / total_examples, 1.0 * avg_rank_dist / total_examples)
        )
        print(
            "avg_num_neg:%.3lf, avg_num_neg_strict:%.3lf\n"
            % (1.0 * avg_num_neg / total_examples, 1.0 * avg_num_neg_strict / total_examples)
        )
        print("occupancy.sizes=%s" % (np.mean(np.array(occupancy_sizes), axis=0)))

    print("save data ...")
    # save label dict
    label_file = open(os.path.join(save_folder, "label_dict.pkl"), "wb")
    pickle.dump(label_dict, label_file)
    label_file.close()

    # save preprocessed data
    all_file = open(os.path.join(save_folder, "all.data.pkl"), "wb")
    pickle.dump(preprocessed_data, all_file)
    all_file.close()

    # split data (80 / 10 / 10)
    random.shuffle(preprocessed_data)
    train_split = int(len(preprocessed_data) * 0.8)
    val_split = int(len(preprocessed_data) * 0.9)
    train_data = preprocessed_data[:train_split]
    val_data = preprocessed_data[train_split:val_split]
    test_data = preprocessed_data[val_split:]

    def save_data(data, data_type):
        f = open(os.path.join(save_folder, "%s.data.pkl" % data_type), "wb")
        pickle.dump(data, f)
        f.close()

    save_data(train_data, "train")
    save_data(val_data, "val")
    save_data(test_data, "test")

    print("finish preprocessing ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-load_folder",
        "--load_folder",
        type=str,
        help="Raw Data Folder",
        required=False,
        default="../../../house_segment/order_data/",
    )
    parser.add_argument(
        "-save_folder",
        "--save_folder",
        type=str,
        help="Save Data Folder",
        required=False,
        default="../../../house_segment/order_data/",
    )
    parser.add_argument(
        "-seed", "--seed", type=int, help="Programwide random seed", required=False, default=9
    )
    parser.add_argument(
        "-debug", "--debug", type=bool, help="Debug Mode", required=False, default=False
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.debug:
        args.save_folder = os.path.join(args.save_folder, "debug2")
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    print("data preprocessing ...")
    data_preprocess(args.load_folder, args.save_folder, args.debug)
    print("done ...")

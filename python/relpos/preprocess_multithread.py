""" Preprocessing """
import os
import argparse
import random
import pickle
import numpy as np
import glob
from IPython import embed

# from tqdm import tqdm
from multiprocessing import Pool


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


def get_negative_examples(occupancy, xyzs, at_most=20):
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
    return negative_examples[: min(len(negative_examples), at_most)]


label_dict = {}


def data_preprocess_per_file(raw_file):
    if True:
        print("raw_file=", raw_file)
        raw_data = np.load(raw_file)

        # first find maxx, maxy, maxz
        inf = 10000
        maxx, maxy, maxz = -inf, -inf, -inf
        minx, miny, minz = inf, inf, inf
        assert len(raw_data) > 0
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
        for c, component in enumerate(raw_data):
            label, xyzs = component
            # get label id
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
    return house_data


def data_preprocess(load_folder, save_folder, debug=False):
    """
        Given components sorted by time heuristic,
            calculate corresponding centroid, positive/negative examples
    """
    raw_files = glob.glob(load_folder + "*.pkl")
    print("load %d raw data files ..." % len(raw_files))
    global label_dict
    label_dict = {}
    # get label dict
    filter_raw_files = []
    for i, raw_file in enumerate(raw_files):
        raw_data = np.load(raw_file)
        if len(raw_data) == 0:
            continue
        filter_raw_files.append(raw_file)
        for component in raw_data:
            label, xyzs = component
            if label not in label_dict:
                label_dict[label] = len(label_dict)
    if debug:
        filter_raw_files = filter_raw_files[:32]
    # preprocessed_data
    print("start preprocessing ...")
    print("label dict:", label_dict)
    with Pool(40) as p:
        preprocessed_data = p.map(data_preprocess_per_file, filter_raw_files)
    print("preprocessed data ... total: %d" % len(preprocessed_data))
    if debug:
        embed()

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
        args.save_folder = os.path.join(args.save_folder, "debug")
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    else:
        args.save_folder = os.path.join(args.save_folder, "real")
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
    print("data preprocessing ...")
    data_preprocess(args.load_folder, args.save_folder, args.debug)
    print("done ...")

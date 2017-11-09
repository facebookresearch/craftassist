#!/usr/bin/python

from autocorrect import spell
import numpy as np
import pickle
import random
import operator
import multiprocessing
import os
import re
import sys

sys.path.append("../..")
import util


def divide_inst(inst):
    visited = set()
    insts = []

    def _bfs(xyz):
        q = [xyz]
        visited.add(xyz)
        ret = []
        while q:
            cur = q.pop(0)
            ret.append(cur)
            neighbors = util.adjacent(cur)
            for n in neighbors:
                if n in inst and n not in visited:
                    visited.add(n)
                    q.append(n)
        return ret

    for xyz in inst:
        if xyz not in visited:
            insts.append(_bfs(xyz))

    return insts


def convert_insts_to_npy(schematic, insts):
    ## regions corresponding to null are 0s
    inst_anno = np.zeros(schematic.shape[:-1])  # less an id dimension
    inst_cls = ["nothing"]
    inst_id = 0
    for label, inst in insts:
        divided_insts = divide_inst(inst)
        for locs in divided_insts:
            inst_id += 1
            inst_cls.append(label)
            for l in locs:
                inst_anno[l[0], l[1], l[2]] = inst_id

    return inst_anno, inst_cls


def read_schematic(house_name):
    ## first read the house blocks
    house_file = os.path.expanduser("~") + "/minecraft_houses/%s/schematic.npy" % house_name
    schematic = np.load(house_file)
    ## y-z-x -> x-y-z
    schematic = np.swapaxes(np.swapaxes(schematic, 1, 2), 0, 1)
    ## each element of cube is [id, mid]
    return schematic


def parse_labels(labels):
    tokens = re.split("\(|\)", labels)
    cnt = 0
    ret = []
    while cnt < len(tokens) - 1:
        label = spell(tokens[cnt].strip())
        i = 1
        # sometimes the label might include "(...)"
        while True:
            try:
                n = int(tokens[cnt + i])
                break
            except:
                label += " ({})".format(tokens[cnt + i].strip())
                i += 1
        cnt += i + 1
        ret.append((label, n))
    return ret


def get_insts_anno(schematic, actions, labels):
    tokens = re.split(",|:", actions)
    # tokens[i] -> " (x y z)"
    xyzs = [tuple(map(int, tokens[i].strip()[1:-1].split())) for i in range(0, len(tokens) - 1, 2)]
    xyzs = [webcraft_xyz_to_minecraft_xyz(schematic, *xyz) for xyz in xyzs]
    offset = 0
    insts = []
    for label, n in labels:
        insts.append((label, xyzs[offset : offset + n]))
        offset += n
    assert offset == len(xyzs), "Number inconsistent!"
    return insts


def webcraft_xyz_to_minecraft_xyz(schematic, x, y, z):
    world_size = 64
    Y, Z, X = schematic.shape[:-1]
    x_offset = int(world_size / 2 - X / 2)
    y_offset = int(world_size / 2 - Y / 2)
    z_offset = 3  # ground height

    x -= x_offset
    y -= y_offset
    z -= z_offset
    assert x >= 0 and y >= 0 and z >= 0, "Coordinates error!"
    assert x < X and y < Y and z < Z, "Coordinates error!"

    return y, z, x


if __name__ == "__main__":
    with open("results.csv", "r") as f:
        lines = f.read().splitlines()

    with open("iframes.csv", "r") as f:
        iframes = set(f.read().splitlines()[1:])

    keys = lines[0].split('"')
    actions_idx = keys.index("Answer.actions")
    labels_idx = keys.index("Answer.labels")
    iframe_idx = keys.index("Input.iframe_url")

    data = []
    annotated_iframes = set()

    def process_line(l):
        tokens = l.split('"')
        labels = tokens[labels_idx].lower()
        res = parse_labels(labels)
        return res

    pool = multiprocessing.Pool(40)
    nested_labels = pool.map(process_line, lines[1:])
    pool.close()
    pool.join()
    labels = [l for ls in nested_labels for l in ls]
    tags = {}
    for l, n in labels:
        if l not in tags:
            tags[l] = n
        else:
            tags[l] += n

    tags = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)

    for i, l in enumerate(lines[1:]):
        tokens = l.split('"')
        actions = tokens[actions_idx]
        labels = nested_labels[i]
        iframe = tokens[iframe_idx]
        if iframe not in iframes:
            continue

        annotated_iframes.add(iframe)

        house_name = os.path.basename(iframe).replace(".html", "")
        schematic = read_schematic(house_name)
        insts = get_insts_anno(schematic, actions, labels)
        if insts:
            inst_anno, inst_cls = convert_insts_to_npy(schematic, insts)
            data.append((schematic[:, :, :, 0], inst_anno, inst_cls, house_name))
        else:
            print("Warning: no instance for {}".format(house_name))

    print(len(data))
    assert iframes == annotated_iframes

    training_ratio = 0.9
    random.shuffle(data)
    training_data = data[: int(training_ratio * len(data))]
    validation_data = data[len(training_data) :]

    with open("./training_data.pkl", "wb") as f:
        pickle.dump(training_data, f)
    with open("./validation_data.pkl", "wb") as f:
        pickle.dump(validation_data, f)

    # # write some examples
    # examples = random.sample(training_data, 10)
    # examples = [(e[1], e[3]) for e in examples]
    # for e, name in examples:
    #     npy = np.zeros((e.shape[0], e.shape[1], e.shape[2], 2))
    #     npy[:, :, :, 0] = e
    #     npy = np.expand_dims(npy, axis=-1)
    #     npy = np.swapaxes(np.swapaxes(npy, 0, 2), 0, 1)
    #     np.save("./inst_examples/{}.npy".format(name), npy)

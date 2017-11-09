import numpy as np
import sys

sys.path.append("..")
from block_data import BLOCK_GROUPS, NORMAL_BLOCKS_N

voxel_group = False
random_embedding_table = np.random.rand(2268, 16)


def random_embedding(id):
    return random_embedding_table[id]


def compute_id_mapping():
    id_mapping = {}
    group_id = {}
    ids = list(range(NORMAL_BLOCKS_N)) + BLOCK_GROUPS["disc"]
    counter = 0
    for id in ids:
        for k, l in BLOCK_GROUPS.items():
            if id in set(l):
                if k not in group_id:
                    group_id[k] = counter
                    counter += 1
                id_mapping[id] = group_id[k]
                break
        else:
            id_mapping[id] = counter
            counter += 1
    return counter, id_mapping


total_id_groups_n, id_mapping = compute_id_mapping()


def check_block_group_id(blocks):
    if hasattr(blocks, "__len__"):
        shape = blocks.shape
        blocks = blocks.reshape(-1)
        for i in range(blocks.size):
            blocks[i] = id_mapping[blocks[i]]
        return blocks.reshape(shape)
    else:  ## scalar
        return id_mapping[blocks]


total_ids_single_n = NORMAL_BLOCKS_N + 1


def check_block_single_id(blocks):
    if hasattr(blocks, "__len__"):
        blocks[blocks >= NORMAL_BLOCKS_N] = NORMAL_BLOCKS_N
        return blocks
    else:
        return NORMAL_BLOCKS_N if blocks >= NORMAL_BLOCKS_N else blocks


def check_block_id():
    if voxel_group:
        return check_block_group_id
    else:
        return check_block_single_id


def total_ids_n():
    if voxel_group:
        return total_id_groups_n
    else:
        return total_ids_single_n

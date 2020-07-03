import pickle
import argparse
import glob
import os

import numpy as np

from typing import List, Dict, Set, Tuple
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm


def open_house_schematic(house_directory: Path) -> np.ndarray:
    with open(Path(house_directory) / "schematic.npy", "rb") as file:
        return np.load(file)


def get_unique_pairs(house_dir: Path) -> Set:
    try:
        pairs = set()
        schematic = open_house_schematic(house_dir)
        
        # House schematic is in yzx format
        # (instance schematics are in xyz).
        for y in range(schematic.shape[0]):
            for z in range(schematic.shape[1]):
                for x in range(schematic.shape[2]):
                    pair = (int(schematic[y, z, x, 0]), int(schematic[y, z, x, 1]))
                    pairs.add(pair)
        
        return pairs
    except FileNotFoundError:
        print(f"schematic not found at {house_dir}")
        return set()



def make_id_vocabulary(houses_data_path: Path) -> Dict[Tuple[int, int], int]:
    
    all_houses = glob.glob(str(houses_data_path / "houses" / "*"))

    all_sets = map(get_unique_pairs, tqdm(all_houses))
        
    block_meta_pairs: Set[Tuple[int, int]] = set()
    for s in all_sets:
        block_meta_pairs = block_meta_pairs.union(s)
    
    vocabulary = {pair: i for i, pair in enumerate(block_meta_pairs)}
    return vocabulary


def make_new_item(house_data_path: Path, item: List, vocabulary) -> List:
    instance_schematic: np.ndarray = item[0]
    house_name = item[-1]
    house_schematic = open_house_schematic(house_data_path / "houses" / house_name)


    assert house_schematic.shape[0] == instance_schematic.shape[1]
    assert house_schematic.shape[1] == instance_schematic.shape[2]
    assert house_schematic.shape[2] == instance_schematic.shape[0]

    new_schematic = instance_schematic.astype(np.int16)

    for x in range(instance_schematic.shape[0]):
        for y in range(instance_schematic.shape[1]):
            for z in range(instance_schematic.shape[2]):
                pair = (int(house_schematic[y, z, x, 0]), int(house_schematic[y, z, x, 1]))
                new_schematic[x, y, z] = vocabulary[pair]
    
    new_item = list(deepcopy(item))
    new_item[0] = new_schematic
    return tuple(new_item)

def create_new_seg_ds(house_data_path: Path, segmentation_data: List, vocabulary: Dict[Tuple[int, int], int]) -> List:
    new_seg_data = []

    for item in segmentation_data:
        new_item = make_new_item(house_data_path, item, vocabulary)
        new_seg_data.append(new_item)

    return new_seg_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-data-dir", "--house", type=str, required=True)
    parser.add_argument("--segmentation-data-dir", "--seg", type=str, required=True)
    parser.add_argument("--out-dir", "--out", type=str, required=True)
    parser.add_argument("--vocabulary-in", "--vin", type=str, required=False)
    parser.add_argument("--vocabulary-out", "--vout", type=str, required=False)

    args = parser.parse_args()

    assert args.vocabulary_in is not None or args.vocabulary_out is not None, "Must specify vin or vout"

    house_data_dir = Path(args.house_data_dir)
    segmentation_data_dir = Path(args.segmentation_data_dir)
    out_dir = Path(args.out_dir)

    os.makedirs(out_dir, exist_ok=True)

    vocab_in = args.vocabulary_in

    if vocab_in:
        with open(vocab_in, "rb") as file:
            vocabulary = pickle.load(file)
    
    else:
        vocabulary = make_id_vocabulary(house_data_dir)
        with open(args.vocabulary_out, "wb") as file:
            pickle.dump(vocabulary, file)

    for ds_name in ["training_data.pkl", "validation_data.pkl"]:
        in_path = segmentation_data_dir / ds_name
        out_path = out_dir / ds_name

        with open(in_path, "rb") as file:
            seg_data = pickle.load(file)

        new_ds = create_new_seg_ds(house_data_dir, seg_data, vocabulary)

        with open(out_path, "wb") as file:
            pickle.dump(new_ds, file)



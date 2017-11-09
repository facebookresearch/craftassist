import logging
import pickle
import numpy as np
import operator
import multiprocessing
from voxel_cube import get_voxel_center_cube, make_center
from local_voxel_cnn import LocalVoxelCNN
from abc import ABC, abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, topk):
        """
        Load cubes of cube_size with the annotated tags.
        """
        pass


class CenterVoxelCubeLoader(DataLoader):
    def __init__(self):
        self.tag_weights = None
        self.cuda_tag_weights = None

    def _cube_f(self, anno):
        schematic, t, loc, cube_size = anno
        return (t, get_voxel_center_cube(loc, schematic, cube_size))

    def load_data(self, topk):
        def read_tag_cubes(x, pkl):
            with open(pkl, "rb") as f:
                y = pickle.load(f)

            annos = []
            for schematic, inst_anno, inst_cls, _ in y:
                idx = zip(*np.nonzero(schematic))
                for loc in idx:
                    inst_id = int(inst_anno[loc[0], loc[1], loc[2]])
                    annos.append(
                        (
                            np.expand_dims(schematic, -1),
                            inst_cls[inst_id],
                            loc,
                            LocalVoxelCNN.cube_size,
                        )
                    )

            pool = multiprocessing.Pool(40)
            cubes = pool.map(self._cube_f, annos)
            pool.close()
            pool.join()
            for t, cube in cubes:
                if t in x:
                    x[t].append(cube)
                else:
                    x[t] = [cube]

        def merge_tags(tags_to_merge, cubes):
            merge_data = [c for ml in tags_to_merge for c in cubes.get(ml, [])]
            for ml in tags_to_merge:
                cubes.pop(ml, None)
            cubes["nothing"] = merge_data

        training_cubes, validation_cubes = {}, {}
        for d in ["training3"]:
            read_tag_cubes(training_cubes, d + "/training_data.pkl")
            read_tag_cubes(validation_cubes, d + "/validation_data.pkl")

        tags = set(training_cubes.keys()) | set(validation_cubes.keys())
        tags = sorted(
            tags, key=lambda t: -len(training_cubes.get(t, []) + validation_cubes.get(t, []))
        )

        tags_to_merge = tags[topk:] + ["none"]
        merge_tags(tags_to_merge, training_cubes)
        merge_tags(tags_to_merge, validation_cubes)
        tags = tags[:topk]
        assert "none" in tags
        tags.remove("none")
        tags.append("nothing")

        tags_n = [len(training_cubes.get(t, []) + validation_cubes.get(t, [])) for t in tags]
        tags = list(zip(tags, tags_n))

        logging.info("Tags: {}".format(tags))

        ## load training and validation data
        training_data = []
        validation_data = []
        for k, (t, _) in enumerate(tags):
            td = training_cubes[t]
            vd = validation_cubes.get(t, [])
            if not vd:
                print("Missing in validation: " + t)
            for i, d in enumerate(td + vd):
                if i < len(td):
                    data = training_data
                else:
                    data = validation_data
                data.append((d, np.array([k]).astype(np.int64)))

        return training_data, validation_data, tags


class GlobalVoxelLoader(DataLoader):
    def load_data(self, topk):
        with open("training3/training_data.pkl", "rb") as f:
            training_data = pickle.load(f)
        with open("training3/validation_data.pkl", "rb") as f:
            validation_data = pickle.load(f)

        size = 32
        # WARNING: make_center might erase some voxels if the cube
        #          already has the max size!
        training_data = [
            (
                make_center(td[0], size),  # schematic
                make_center(td[1], size),  # inst_anno
                td[2],
            )  # inst_to_cls mapping
            for td in training_data
        ]
        validation_data = [
            (make_center(vd[0], size), make_center(vd[1], size), vd[2]) for vd in validation_data
        ]

        def _get_tags_statistics(data):
            tags = dict()
            for _, inst_anno, inst_cls in data:
                assert inst_cls[0] == "nothing"
                for i in range(1, len(inst_cls)):  # 0 -> 'nothing'
                    t = inst_cls[i]
                    if t not in tags:
                        tags[t] = 0
                    tags[t] += (inst_anno == i).sum()
            # get tags sorted in the descending order
            tags = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
            return tags

        tags = _get_tags_statistics(training_data + validation_data)

        # merge minor tags and 'none' into 'nothing'
        minor_n = sum([t[1] for t in tags[topk:]])
        tags_to_merge = [t[0] for t in tags[topk:]] + ["none"]

        def _merge_minor_tags(data):
            for _, inst_anno, inst_cls in data:
                for i, t in enumerate(inst_cls):
                    if t in tags_to_merge:
                        idx = np.nonzero(inst_anno == i)
                        inst_anno[idx] = 0

        _merge_minor_tags(training_data)
        _merge_minor_tags(validation_data)

        tags = dict(tags[:topk])
        nothing_n = tags.pop("none") + minor_n
        tags["nothing"] = nothing_n
        tags = list(tags.items())
        tag_keys = [k for k, n in tags]
        print(tags)

        def get_cls_anno(inst_anno, inst_cls):
            cls_anno = np.copy(inst_anno)
            for i, t in enumerate(inst_cls):
                idx = np.nonzero(inst_anno == i)
                if idx[0].size > 0:
                    cls_anno[idx] = tag_keys.index(t)
            return cls_anno

        training_data = [
            (sch, inst_anno, get_cls_anno(inst_anno, inst_cls))
            for sch, inst_anno, inst_cls in training_data
        ]
        validation_data = [
            (sch, inst_anno, get_cls_anno(inst_anno, inst_cls))
            for sch, inst_anno, inst_cls in validation_data
        ]
        return training_data, validation_data, tags
